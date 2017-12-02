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
from logging import getLogger


logger = getLogger()


AVAILABLE_ATTR = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def log_attributes_stats(train_attributes, valid_attributes, test_attributes, params):
    """
    Log attributes distributions.
    """
    k = 0
    for (attr_name, n_cat) in params.attr:
        logger.debug('Train %s: %s' % (attr_name, ' / '.join(['%.5f' % train_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Valid %s: %s' % (attr_name, ' / '.join(['%.5f' % valid_attributes[:, k + i].mean() for i in range(n_cat)])))
        logger.debug('Test  %s: %s' % (attr_name, ' / '.join(['%.5f' % test_attributes[:, k + i].mean() for i in range(n_cat)])))
        assert train_attributes[:, k:k + n_cat].sum() == train_attributes.size(0)
        assert valid_attributes[:, k:k + n_cat].sum() == valid_attributes.size(0)
        assert test_attributes[:, k:k + n_cat].sum() == test_attributes.size(0)
        k += n_cat
    assert k == params.n_attr


def load_images(params):
    """
    Load celebA dataset.
    """
    # load data
    images_filename = 'images_%i_%i_20000.pth' if params.debug else 'images_%i_%i.pth'
    images_filename = images_filename % (params.img_sz, params.img_sz)
    images = torch.load(os.path.join(DATA_PATH, images_filename))
    attributes = torch.load(os.path.join(DATA_PATH, 'attributes.pth'))

    # parse attributes
    attrs = []
    for name, n_cat in params.attr:
        for i in range(n_cat):
            attrs.append(torch.FloatTensor((attributes[name] == i).astype(np.float32)))
    attributes = torch.cat([x.unsqueeze(1) for x in attrs], 1)
    # split train / valid / test
    if params.debug:
        train_index = 10000
        valid_index = 15000
        test_index = 20000
    else:
        train_index = 162770
        valid_index = 162770 + 19867
        test_index = len(images)
    train_images = images[:train_index]
    valid_images = images[train_index:valid_index]
    test_images = images[valid_index:test_index]
    train_attributes = attributes[:train_index]
    valid_attributes = attributes[train_index:valid_index]
    test_attributes = attributes[valid_index:test_index]
    # log dataset statistics / return dataset
    logger.info('%i / %i / %i images with attributes for train / valid / test sets'
                % (len(train_images), len(valid_images), len(test_images)))
    log_attributes_stats(train_attributes, valid_attributes, test_attributes, params)
    images = train_images, valid_images, test_images
    attributes = train_attributes, valid_attributes, test_attributes
    return images, attributes


def normalize_images(images):
    """
    Normalize image values.
    """
    return images.float().div_(255.0).mul_(2.0).add_(-1)


class DataSampler(object):

    def __init__(self, images, attributes, params):
        """
        Initialize the data sampler with training data.
        """
        assert images.size(0) == attributes.size(0), (images.size(), attributes.size())
        self.images = images
        self.attributes = attributes
        self.batch_size = params.batch_size
        self.v_flip = params.v_flip
        self.h_flip = params.h_flip

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        return self.images.size(0)

    def train_batch(self, bs):
        """
        Get a batch of random images with their attributes.
        """
        # image IDs
        idx = torch.LongTensor(bs).random_(len(self.images))

        # select images / attributes
        batch_x = normalize_images(self.images.index_select(0, idx).cuda())
        batch_y = self.attributes.index_select(0, idx).cuda()

        # data augmentation
        if self.v_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(2, torch.arange(batch_x.size(2) - 1, -1, -1).long().cuda())
        if self.h_flip and np.random.rand() <= 0.5:
            batch_x = batch_x.index_select(3, torch.arange(batch_x.size(3) - 1, -1, -1).long().cuda())

        return Variable(batch_x, volatile=False), Variable(batch_y, volatile=False)

    def eval_batch(self, i, j):
        """
        Get a batch of images in a range with their attributes.
        """
        assert i < j
        batch_x = normalize_images(self.images[i:j].cuda())
        batch_y = self.attributes[i:j].cuda()
        return Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
