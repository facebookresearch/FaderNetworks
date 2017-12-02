# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
from logging import getLogger

from .model import update_predictions, flip_attributes
from .utils import print_accuracies


logger = getLogger()


class Evaluator(object):

    def __init__(self, ae, lat_dis, ptc_dis, clf_dis, eval_clf, data, params):
        """
        Evaluator initialization.
        """
        # data / parameters
        self.data = data
        self.params = params

        # modules
        self.ae = ae
        self.lat_dis = lat_dis
        self.ptc_dis = ptc_dis
        self.clf_dis = clf_dis
        self.eval_clf = eval_clf
        assert eval_clf.img_sz == params.img_sz
        assert all(attr in eval_clf.attr for attr in params.attr)

    def eval_reconstruction_loss(self):
        """
        Compute the autoencoder reconstruction perplexity.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        bs = params.batch_size

        costs = []
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            _, dec_outputs = self.ae(batch_x, batch_y)
            costs.append(((dec_outputs[-1] - batch_x) ** 2).mean().data[0])

        return np.mean(costs)

    def eval_lat_dis_accuracy(self):
        """
        Compute the latent discriminator prediction accuracy.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        self.lat_dis.eval()
        bs = params.batch_size

        all_preds = [[] for _ in range(len(params.attr))]
        for i in range(0, len(data), bs):
            batch_x, batch_y = data.eval_batch(i, i + bs)
            enc_outputs = self.ae.encode(batch_x)
            preds = self.lat_dis(enc_outputs[-1 - params.n_skip]).data.cpu()
            update_predictions(all_preds, preds, batch_y.data.cpu(), params)

        return [np.mean(x) for x in all_preds]

    def eval_ptc_dis_accuracy(self):
        """
        Compute the patch discriminator prediction accuracy.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        self.ptc_dis.eval()
        bs = params.batch_size

        real_preds = []
        fake_preds = []

        for i in range(0, len(data), bs):
            # batch / encode / decode
            batch_x, batch_y = data.eval_batch(i, i + bs)
            flipped = flip_attributes(batch_y, params, 'all')
            _, dec_outputs = self.ae(batch_x, flipped)
            # predictions
            real_preds.extend(self.ptc_dis(batch_x).data.tolist())
            fake_preds.extend(self.ptc_dis(dec_outputs[-1]).data.tolist())

        return real_preds, fake_preds

    def eval_clf_dis_accuracy(self):
        """
        Compute the classifier discriminator prediction accuracy.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        self.clf_dis.eval()
        bs = params.batch_size

        all_preds = [[] for _ in range(params.n_attr)]
        for i in range(0, len(data), bs):
            # batch / encode / decode
            batch_x, batch_y = data.eval_batch(i, i + bs)
            enc_outputs = self.ae.encode(batch_x)
            # flip all attributes one by one
            k = 0
            for j, (_, n_cat) in enumerate(params.attr):
                for value in range(n_cat):
                    flipped = flip_attributes(batch_y, params, j, new_value=value)
                    dec_outputs = self.ae.decode(enc_outputs, flipped)
                    # classify
                    clf_dis_preds = self.clf_dis(dec_outputs[-1])[:, j:j + n_cat].max(1)[1].view(-1)
                    all_preds[k].extend((clf_dis_preds.data.cpu() == value).tolist())
                    k += 1
            assert k == params.n_attr

        return [np.mean(x) for x in all_preds]

    def eval_clf_accuracy(self):
        """
        Compute the accuracy of flipped attributes according to the trained classifier.
        """
        data = self.data
        params = self.params
        self.ae.eval()
        bs = params.batch_size

        idx = []
        for j in range(len(params.attr)):
            attr_index = self.eval_clf.attr.index(params.attr[j])
            idx.append(sum([x[1] for x in self.eval_clf.attr[:attr_index]]))

        all_preds = [[] for _ in range(params.n_attr)]
        for i in range(0, len(data), bs):
            # batch / encode / decode
            batch_x, batch_y = data.eval_batch(i, i + bs)
            enc_outputs = self.ae.encode(batch_x)
            # flip all attributes one by one
            k = 0
            for j, (_, n_cat) in enumerate(params.attr):
                for value in range(n_cat):
                    flipped = flip_attributes(batch_y, params, j, new_value=value)
                    dec_outputs = self.ae.decode(enc_outputs, flipped)
                    # classify
                    clf_preds = self.eval_clf(dec_outputs[-1])[:, idx[j]:idx[j] + n_cat].max(1)[1].view(-1)
                    all_preds[k].extend((clf_preds.data.cpu() == value).tolist())
                    k += 1
            assert k == params.n_attr

        return [np.mean(x) for x in all_preds]

    def evaluate(self, n_epoch):
        """
        Evaluate all models / log evaluation results.
        """
        params = self.params
        logger.info('')

        # reconstruction loss
        ae_loss = self.eval_reconstruction_loss()

        # latent discriminator accuracy
        log_lat_dis = []
        if params.n_lat_dis:
            lat_dis_accu = self.eval_lat_dis_accuracy()
            log_lat_dis.append(('lat_dis_accu', np.mean(lat_dis_accu)))
            for accu, (name, _) in zip(lat_dis_accu, params.attr):
                log_lat_dis.append(('lat_dis_accu_%s' % name, accu))
            logger.info('Latent discriminator accuracy:')
            print_accuracies(log_lat_dis)

        # patch discriminator accuracy
        log_ptc_dis = []
        if params.n_ptc_dis:
            ptc_dis_real_preds, ptc_dis_fake_preds = self.eval_ptc_dis_accuracy()
            accu_real = (np.array(ptc_dis_real_preds).astype(np.float32) >= 0.5).mean()
            accu_fake = (np.array(ptc_dis_fake_preds).astype(np.float32) <= 0.5).mean()
            log_ptc_dis.append(('ptc_dis_preds_real', np.mean(ptc_dis_real_preds)))
            log_ptc_dis.append(('ptc_dis_preds_fake', np.mean(ptc_dis_fake_preds)))
            log_ptc_dis.append(('ptc_dis_accu_real', accu_real))
            log_ptc_dis.append(('ptc_dis_accu_fake', accu_fake))
            log_ptc_dis.append(('ptc_dis_accu', (accu_real + accu_fake) / 2))
            logger.info('Patch discriminator accuracy:')
            print_accuracies(log_ptc_dis)

        # classifier discriminator accuracy
        log_clf_dis = []
        if params.n_clf_dis:
            clf_dis_accu = self.eval_clf_dis_accuracy()
            k = 0
            log_clf_dis += [('clf_dis_accu', np.mean(clf_dis_accu))]
            for name, n_cat in params.attr:
                log_clf_dis.append(('clf_dis_accu_%s' % name, np.mean(clf_dis_accu[k:k + n_cat])))
                log_clf_dis.extend([('clf_dis_accu_%s_%i' % (name, j), clf_dis_accu[k + j])
                                    for j in range(n_cat)])
                k += n_cat
            logger.info('Classifier discriminator accuracy:')
            print_accuracies(log_clf_dis)

        # classifier accuracy
        log_clf = []
        clf_accu = self.eval_clf_accuracy()
        k = 0
        log_clf += [('clf_accu', np.mean(clf_accu))]
        for name, n_cat in params.attr:
            log_clf.append(('clf_accu_%s' % name, np.mean(clf_accu[k:k + n_cat])))
            log_clf.extend([('clf_accu_%s_%i' % (name, j), clf_accu[k + j])
                            for j in range(n_cat)])
            k += n_cat
        logger.info('Classifier accuracy:')
        print_accuracies(log_clf)

        # log autoencoder loss
        logger.info('Autoencoder loss: %.5f' % ae_loss)

        # JSON log
        to_log = dict([
            ('n_epoch', n_epoch),
            ('ae_loss', ae_loss)
        ] + log_lat_dis + log_ptc_dis + log_clf_dis + log_clf)
        logger.debug("__log__:%s" % json.dumps(to_log))

        return to_log


def compute_accuracy(classifier, data, params):
    """
    Compute the classifier prediction accuracy.
    """
    classifier.eval()
    bs = params.batch_size

    all_preds = [[] for _ in range(len(classifier.attr))]
    for i in range(0, len(data), bs):
        batch_x, batch_y = data.eval_batch(i, i + bs)
        preds = classifier(batch_x).data.cpu()
        update_predictions(all_preds, preds, batch_y.data.cpu(), params)

    return [np.mean(x) for x in all_preds]
