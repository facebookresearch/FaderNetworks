# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import torch

from src.loader import load_images, DataSampler
from src.utils import initialize_exp, bool_flag, attr_flag, check_attr
from src.model import AutoEncoder, LatentDiscriminator, PatchDiscriminator, Classifier
from src.training import Trainer
from src.evaluation import Evaluator


# parse parameters
parser = argparse.ArgumentParser(description='Images autoencoder')
parser.add_argument("--name", type=str, default="default",
                    help="Experiment name")
parser.add_argument("--img_sz", type=int, default=256,
                    help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3,
                    help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Smiling,Male",
                    help="Attributes to classify")
parser.add_argument("--instance_norm", type=bool_flag, default=False,
                    help="Use instance normalization instead of batch normalization")
parser.add_argument("--init_fm", type=int, default=32,
                    help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512,
                    help="Number maximum of filters in the autoencoder")
parser.add_argument("--n_layers", type=int, default=6,
                    help="Number of layers in the encoder / decoder")
parser.add_argument("--n_skip", type=int, default=0,
                    help="Number of skip connections")
parser.add_argument("--deconv_method", type=str, default="convtranspose",
                    help="Deconvolution method")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension for discriminator / classifier")
parser.add_argument("--dec_dropout", type=float, default=0.,
                    help="Dropout in the decoder")
parser.add_argument("--lat_dis_dropout", type=float, default=0.3,
                    help="Dropout in the latent discriminator")
parser.add_argument("--n_lat_dis", type=int, default=1,
                    help="Number of latent discriminator training steps")
parser.add_argument("--n_ptc_dis", type=int, default=0,
                    help="Number of patch discriminator training steps")
parser.add_argument("--n_clf_dis", type=int, default=0,
                    help="Number of classifier discriminator training steps")
parser.add_argument("--smooth_label", type=float, default=0.2,
                    help="Smooth label for patch discriminator")
parser.add_argument("--lambda_ae", type=float, default=1,
                    help="Autoencoder loss coefficient")
parser.add_argument("--lambda_lat_dis", type=float, default=0.0001,
                    help="Latent discriminator loss feedback coefficient")
parser.add_argument("--lambda_ptc_dis", type=float, default=0,
                    help="Patch discriminator loss feedback coefficient")
parser.add_argument("--lambda_clf_dis", type=float, default=0,
                    help="Classifier discriminator loss feedback coefficient")
parser.add_argument("--lambda_schedule", type=float, default=500000,
                    help="Progressively increase discriminators' lambdas (0 to disable)")
parser.add_argument("--v_flip", type=bool_flag, default=False,
                    help="Random vertical flip for data augmentation")
parser.add_argument("--h_flip", type=bool_flag, default=True,
                    help="Random horizontal flip for data augmentation")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--ae_optimizer", type=str, default="adam,lr=0.0002",
                    help="Autoencoder optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--dis_optimizer", type=str, default="adam,lr=0.0002",
                    help="Discriminator optimizer (SGD / RMSprop / Adam, etc.)")
parser.add_argument("--clip_grad_norm", type=float, default=5,
                    help="Clip gradient norms (0 to disable)")
parser.add_argument("--n_epochs", type=int, default=1000,
                    help="Total number of epochs")
parser.add_argument("--epoch_size", type=int, default=50000,
                    help="Number of samples per epoch")
parser.add_argument("--ae_reload", type=str, default="",
                    help="Reload a pretrained encoder")
parser.add_argument("--lat_dis_reload", type=str, default="",
                    help="Reload a pretrained latent discriminator")
parser.add_argument("--ptc_dis_reload", type=str, default="",
                    help="Reload a pretrained patch discriminator")
parser.add_argument("--clf_dis_reload", type=str, default="",
                    help="Reload a pretrained classifier discriminator")
parser.add_argument("--eval_clf", type=str, default="",
                    help="Load an external classifier for evaluation")
parser.add_argument("--debug", type=bool_flag, default=False,
                    help="Debug mode (only load a subset of the whole dataset)")
params = parser.parse_args()

# check parameters
check_attr(params)
assert len(params.name.strip()) > 0
assert params.n_skip <= params.n_layers - 1
assert params.deconv_method in ['convtranspose', 'upsampling', 'pixelshuffle']
assert 0 <= params.smooth_label < 0.5
assert not params.ae_reload or os.path.isfile(params.ae_reload)
assert not params.lat_dis_reload or os.path.isfile(params.lat_dis_reload)
assert not params.ptc_dis_reload or os.path.isfile(params.ptc_dis_reload)
assert not params.clf_dis_reload or os.path.isfile(params.clf_dis_reload)
assert os.path.isfile(params.eval_clf)
assert params.lambda_lat_dis == 0 or params.n_lat_dis > 0
assert params.lambda_ptc_dis == 0 or params.n_ptc_dis > 0
assert params.lambda_clf_dis == 0 or params.n_clf_dis > 0

# initialize experiment / load dataset
logger = initialize_exp(params)
data, attributes = load_images(params)
train_data = DataSampler(data[0], attributes[0], params)
valid_data = DataSampler(data[1], attributes[1], params)

# build the model
ae = AutoEncoder(params).cuda()
lat_dis = LatentDiscriminator(params).cuda() if params.n_lat_dis else None
ptc_dis = PatchDiscriminator(params).cuda() if params.n_ptc_dis else None
clf_dis = Classifier(params).cuda() if params.n_clf_dis else None
eval_clf = torch.load(params.eval_clf).cuda().eval()

# trainer / evaluator
trainer = Trainer(ae, lat_dis, ptc_dis, clf_dis, train_data, params)
evaluator = Evaluator(ae, lat_dis, ptc_dis, clf_dis, eval_clf, valid_data, params)


for n_epoch in range(params.n_epochs):

    logger.info('Starting epoch %i...' % n_epoch)

    for n_iter in range(0, params.epoch_size, params.batch_size):

        # latent discriminator training
        for _ in range(params.n_lat_dis):
            trainer.lat_dis_step()

        # patch discriminator training
        for _ in range(params.n_ptc_dis):
            trainer.ptc_dis_step()

        # classifier discriminator training
        for _ in range(params.n_clf_dis):
            trainer.clf_dis_step()

        # autoencoder training
        trainer.autoencoder_step()

        # print training statistics
        trainer.step(n_iter)

    # run all evaluations / save best or periodic model
    to_log = evaluator.evaluate(n_epoch)
    trainer.save_best_periodic(to_log)
    logger.info('End of epoch %i.\n' % n_epoch)
