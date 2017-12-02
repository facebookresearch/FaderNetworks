# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import json
import argparse
import numpy as np
import torch

from src.loader import load_images, DataSampler
from src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from src.utils import get_optimizer, reload_model, print_accuracies
from src.model import Classifier
from src.training import classifier_step
from src.evaluation import compute_accuracy


# parse parameters
parser = argparse.ArgumentParser(description='Classifier')
parser.add_argument("--name", type=str, default="default",
                    help="Experiment name")
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3,
                    help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Smiling",
                    help="Attributes to classify")
parser.add_argument("--init_fm", type=int, default=32,
                    help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512,
                    help="Number maximum of filters in the autoencoder")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension")
parser.add_argument("--v_flip", type=bool_flag, default=False,
                    help="Random vertical flip for data augmentation")
parser.add_argument("--h_flip", type=bool_flag, default=True,
                    help="Random horizontal flip for data augmentation")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--optimizer", type=str, default="adam",
                    help="Classifier optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradient norms (0 to disable)")
parser.add_argument("--n_epochs", type=int, default=1000,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=50000,
                    help="Number of samples per epoch")
parser.add_argument("--reload", type=str, default="",
                    help="Reload a pretrained classifier")
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only load a subset of the whole dataset)")
params = parser.parse_args()

# check parameters
check_attr(params)
assert len(params.name.strip()) > 0
assert not params.reload or os.path.isfile(params.reload)

# initialize experiment / load dataset
logger = initialize_exp(params)
data, attributes = load_images(params)
train_data = DataSampler(data[0], attributes[0], params)
valid_data = DataSampler(data[1], attributes[1], params)
test_data = DataSampler(data[2], attributes[2], params)

# build the model / reload / optimizer
classifier = Classifier(params).cuda()
if params.reload:
    reload_model(classifier, params.reload,
                 ['img_sz', 'img_fm', 'init_fm', 'hid_dim', 'attr', 'n_attr'])
optimizer = get_optimizer(classifier, params.optimizer)


def save_model(name):
    """
    Save the model.
    """
    path = os.path.join(params.dump_path, '%s.pth' % name)
    logger.info('Saving the classifier to %s ...' % path)
    torch.save(classifier, path)


# best accuracy
best_accu = -1e12


for n_epoch in range(params.n_epochs):

    logger.info('Starting epoch %i...' % n_epoch)
    costs = []

    classifier.train()

    for n_iter in range(0, params.epoch_size, params.batch_size):

        # classifier training
        classifier_step(classifier, optimizer, train_data, params, costs)

        # average loss
        if len(costs) >= 25:
            logger.info('%06i - Classifier loss: %.5f' % (n_iter, np.mean(costs)))
            del costs[:]

    # compute accuracy
    valid_accu = compute_accuracy(classifier, valid_data, params)
    test_accu = compute_accuracy(classifier, test_data, params)

    # log classifier accuracy
    log_accu = [('valid_accu', np.mean(valid_accu)), ('test_accu', np.mean(test_accu))]
    for accu, (name, _) in zip(valid_accu, params.attr):
        log_accu.append(('valid_accu_%s' % name, accu))
    for accu, (name, _) in zip(test_accu, params.attr):
        log_accu.append(('test_accu_%s' % name, accu))
    logger.info('Classifier accuracy:')
    print_accuracies(log_accu)

    # JSON log
    logger.debug("__log__:%s" % json.dumps(dict([('n_epoch', n_epoch)] + log_accu)))

    # save best or periodic model
    if np.mean(valid_accu) > best_accu:
        best_accu = np.mean(valid_accu)
        logger.info('Best validation average accuracy: %.5f' % best_accu)
        save_model('best')
    elif n_epoch % 10 == 0 and n_epoch > 0:
        save_model('periodic-%i' % n_epoch)

    logger.info('End of epoch %i.\n' % n_epoch)
