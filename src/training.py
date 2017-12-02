# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from logging import getLogger

from .utils import get_optimizer, clip_grad_norm, get_lambda, reload_model
from .model import get_attr_loss, flip_attributes


logger = getLogger()


class Trainer(object):

    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, data, params):
        """
        Trainer initialization.
        """
        # data / parameters
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis

        # optimizers
        self.ae_optimizer = get_optimizer(ae, params.ae_optimizer)
        logger.info(ae)
        logger.info('%i parameters in the autoencoder. '
                    % sum([p.nelement() for p in ae.parameters()]))
        if params.n_lat_dis:
            logger.info(lat_dis)
            logger.info('%i parameters in the latent discriminator. '
                        % sum([p.nelement() for p in lat_dis.parameters()]))
            self.lat_dis_optimizer = get_optimizer(lat_dis, params.dis_optimizer)
        if params.n_ptc_dis:
            logger.info(ptc_dis)
            logger.info('%i parameters in the patch discriminator. '
                        % sum([p.nelement() for p in ptc_dis.parameters()]))
            self.ptc_dis_optimizer = get_optimizer(ptc_dis, params.dis_optimizer)
        if params.n_clf_dis:
            logger.info(clf_dis)
            logger.info('%i parameters in the classifier discriminator. '
                        % sum([p.nelement() for p in clf_dis.parameters()]))
            self.clf_dis_optimizer = get_optimizer(clf_dis, params.dis_optimizer)

        # reload pretrained models
        if params.ae_reload:
            reload_model(ae, params.ae_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'n_layers', 'n_skip', 'attr', 'n_attr'])
        if params.lat_dis_reload:
            reload_model(lat_dis, params.lat_dis_reload,
                         ['enc_dim', 'attr', 'n_attr'])
        if params.ptc_dis_reload:
            reload_model(ptc_dis, params.ptc_dis_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'max_fm', 'n_patch_dis_layers'])
        if params.clf_dis_reload:
            reload_model(clf_dis, params.clf_dis_reload,
                         ['img_sz', 'img_fm', 'init_fm', 'max_fm', 'hid_dim', 'attr', 'n_attr'])

        # training statistics
        self.stats = {}
        self.stats['rec_costs'] = []
        self.stats['lat_dis_costs'] = []
        self.stats['ptc_dis_costs'] = []
        self.stats['clf_dis_costs'] = []

        # best reconstruction loss / best accuracy
        self.best_loss = 1e12
        self.best_accu = -1e12
        self.params.n_total_iter = 0

    def lat_dis_step(self):
        """
        Train the latent discriminator.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        self.lat_dis.train()
        bs = params.batch_size
        # batch / encode / discriminate
        batch_x, batch_y = data.train_batch(bs)
        enc_outputs = self.ae.encode(Variable(batch_x.data, volatile=True))
        preds = self.lat_dis(Variable(enc_outputs[-1 - params.n_skip].data))
        # loss / optimize
        loss = get_attr_loss(preds, batch_y, False, params)
        self.stats['lat_dis_costs'].append(loss.data[0])
        self.lat_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.lat_dis.parameters(), params.clip_grad_norm)
        self.lat_dis_optimizer.step()

    def ptc_dis_step(self):
        """
        Train the patch discriminator.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        self.ptc_dis.train()
        bs = params.batch_size
        # batch / encode / discriminate
        batch_x, batch_y = data.train_batch(bs)
        flipped = flip_attributes(batch_y, params, 'all')
        _, dec_outputs = self.ae(Variable(batch_x.data, volatile=True), flipped)
        real_preds = self.ptc_dis(batch_x)
        fake_preds = self.ptc_dis(Variable(dec_outputs[-1].data))
        y_fake = Variable(torch.FloatTensor(real_preds.size())
                               .fill_(params.smooth_label).cuda())
        # loss / optimize
        loss = F.binary_cross_entropy(real_preds, 1 - y_fake)
        loss += F.binary_cross_entropy(fake_preds, y_fake)
        self.stats['ptc_dis_costs'].append(loss.data[0])
        self.ptc_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.ptc_dis.parameters(), params.clip_grad_norm)
        self.ptc_dis_optimizer.step()

    def clf_dis_step(self):
        """
        Train the classifier discriminator.
        """
        data = self.data
        params = self.params
        self.clf_dis.train()
        bs = params.batch_size
        # batch / predict
        batch_x, batch_y = data.train_batch(bs)
        preds = self.clf_dis(batch_x)
        # loss / optimize
        loss = get_attr_loss(preds, batch_y, False, params)
        self.stats['clf_dis_costs'].append(loss.data[0])
        self.clf_dis_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.clf_dis.parameters(), params.clip_grad_norm)
        self.clf_dis_optimizer.step()

    def autoencoder_step(self):
        """
        Train the autoencoder with cross-entropy loss.
        Train the encoder with discriminator loss.
        """
        data = self.data
        params = self.params
        self.ae.train()
        if params.n_lat_dis:
            self.lat_dis.eval()
        if params.n_ptc_dis:
            self.ptc_dis.eval()
        if params.n_clf_dis:
            self.clf_dis.eval()
        bs = params.batch_size
        # batch / encode / decode
        batch_x, batch_y = data.train_batch(bs)
        enc_outputs, dec_outputs = self.ae(batch_x, batch_y)
        # autoencoder loss from reconstruction
        loss = params.lambda_ae * ((batch_x - dec_outputs[-1]) ** 2).mean()
        self.stats['rec_costs'].append(loss.data[0])
        # encoder loss from the latent discriminator
        if params.lambda_lat_dis:
            lat_dis_preds = self.lat_dis(enc_outputs[-1 - params.n_skip])
            lat_dis_loss = get_attr_loss(lat_dis_preds, batch_y, True, params)
            loss = loss + get_lambda(params.lambda_lat_dis, params) * lat_dis_loss
        # decoding with random labels
        if params.lambda_ptc_dis + params.lambda_clf_dis > 0:
            flipped = flip_attributes(batch_y, params, 'all')
            dec_outputs_flipped = self.ae.decode(enc_outputs, flipped)
        # autoencoder loss from the patch discriminator
        if params.lambda_ptc_dis:
            ptc_dis_preds = self.ptc_dis(dec_outputs_flipped[-1])
            y_fake = Variable(torch.FloatTensor(ptc_dis_preds.size())
                                   .fill_(params.smooth_label).cuda())
            ptc_dis_loss = F.binary_cross_entropy(ptc_dis_preds, 1 - y_fake)
            loss = loss + get_lambda(params.lambda_ptc_dis, params) * ptc_dis_loss
        # autoencoder loss from the classifier discriminator
        if params.lambda_clf_dis:
            clf_dis_preds = self.clf_dis(dec_outputs_flipped[-1])
            clf_dis_loss = get_attr_loss(clf_dis_preds, flipped, False, params)
            loss = loss + get_lambda(params.lambda_clf_dis, params) * clf_dis_loss
        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()
        # optimize
        self.ae_optimizer.zero_grad()
        loss.backward()
        if params.clip_grad_norm:
            clip_grad_norm(self.ae.parameters(), params.clip_grad_norm)
        self.ae_optimizer.step()

    def step(self, n_iter):
        """
        End training iteration / print training statistics.
        """
        # average loss
        if len(self.stats['rec_costs']) >= 25:
            mean_loss = [
                ('Latent discriminator', 'lat_dis_costs'),
                ('Patch discriminator', 'ptc_dis_costs'),
                ('Classifier discriminator', 'clf_dis_costs'),
                ('Reconstruction loss', 'rec_costs'),
            ]
            logger.info(('%06i - ' % n_iter) +
                        ' / '.join(['%s : %.5f' % (a, np.mean(self.stats[b]))
                                    for a, b in mean_loss if len(self.stats[b]) > 0]))
            del self.stats['rec_costs'][:]
            del self.stats['lat_dis_costs'][:]
            del self.stats['ptc_dis_costs'][:]
            del self.stats['clf_dis_costs'][:]

        self.params.n_total_iter += 1

    def save_model(self, name):
        """
        Save the model.
        """
        def save(model, filename):
            path = os.path.join(self.params.dump_path, '%s_%s.pth' % (name, filename))
            logger.info('Saving %s to %s ...' % (filename, path))
            torch.save(model, path)
        save(self.ae, 'ae')
        if self.params.n_lat_dis:
            save(self.lat_dis, 'lat_dis')
        if self.params.n_ptc_dis:
            save(self.ptc_dis, 'ptc_dis')
        if self.params.n_clf_dis:
            save(self.clf_dis, 'clf_dis')

    def save_best_periodic(self, to_log):
        """
        Save the best models / periodically save the models.
        """
        if to_log['ae_loss'] < self.best_loss:
            self.best_loss = to_log['ae_loss']
            logger.info('Best reconstruction loss: %.5f' % self.best_loss)
            self.save_model('best_rec')
        if self.params.eval_clf and np.mean(to_log['clf_accu']) > self.best_accu:
            self.best_accu = np.mean(to_log['clf_accu'])
            logger.info('Best evaluation accuracy: %.5f' % self.best_accu)
            self.save_model('best_accu')
        if to_log['n_epoch'] % 5 == 0 and to_log['n_epoch'] > 0:
            self.save_model('periodic-%i' % to_log['n_epoch'])


def classifier_step(classifier, optimizer, data, params, costs):
    """
    Train the classifier.
    """
    classifier.train()
    bs = params.batch_size

    # batch / classify
    batch_x, batch_y = data.train_batch(bs)
    preds = classifier(batch_x)
    # loss / optimize
    loss = get_attr_loss(preds, batch_y, False, params)
    costs.append(loss.data[0])
    optimizer.zero_grad()
    loss.backward()
    if params.clip_grad_norm:
        clip_grad_norm(classifier.parameters(), params.clip_grad_norm)
    optimizer.step()
