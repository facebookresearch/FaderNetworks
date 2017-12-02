# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import re
import pickle
import random
import inspect
import argparse
import subprocess
import torch
from torch import optim
from logging import getLogger

from .logger import create_logger
from .loader import AVAILABLE_ATTR


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


logger = getLogger()


def initialize_exp(params):
    """
    Experiment initialization.
    """
    # dump parameters
    params.dump_path = get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    return logger


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def attr_flag(s):
    """
    Parse attributes parameters.
    """
    if s == "*":
        return s
    attr = s.split(',')
    assert len(attr) == len(set(attr))
    attributes = []
    for x in attr:
        if '.' not in x:
            attributes.append((x, 2))
        else:
            split = x.split('.')
            assert len(split) == 2 and len(split[0]) > 0
            assert split[1].isdigit() and int(split[1]) >= 2
            attributes.append((split[0], int(split[1])))
    return sorted(attributes, key=lambda x: (x[1], x[0]))


def check_attr(params):
    """
    Check attributes validy.
    """
    if params.attr == '*':
        params.attr = attr_flag(','.join(AVAILABLE_ATTR))
    else:
        assert all(name in AVAILABLE_ATTR and n_cat >= 2 for name, n_cat in params.attr)
    params.n_attr = sum([n_cat for _, n_cat in params.attr])


def get_optimizer(model, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.5), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(model.parameters(), **optim_params)


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """
    parameters = list(parameters)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef >= 1:
        return
    for p in parameters:
        p.grad.data.mul_(clip_coef)


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    assert os.path.isdir(MODELS_PATH)

    # create the sweep path if it does not exist
    sweep_path = os.path.join(MODELS_PATH, params.name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir %s" % sweep_path, shell=True).wait()

    # create a random name for the experiment
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    while True:
        exp_id = ''.join(random.choice(chars) for _ in range(10))
        dump_path = os.path.join(MODELS_PATH, params.name, exp_id)
        if not os.path.isdir(dump_path):
            break

    # create the dump folder
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir %s" % dump_path, shell=True).wait()
    return dump_path


def reload_model(model, to_reload, attributes=None):
    """
    Reload a previously trained model.
    """
    # reload the model
    assert os.path.isfile(to_reload)
    to_reload = torch.load(to_reload)

    # check parameters sizes
    model_params = set(model.state_dict().keys())
    to_reload_params = set(to_reload.state_dict().keys())
    assert model_params == to_reload_params, (model_params - to_reload_params,
                                              to_reload_params - model_params)

    # check attributes
    attributes = [] if attributes is None else attributes
    for k in attributes:
        if getattr(model, k, None) is None:
            raise Exception('Attribute "%s" not found in the current model' % k)
        if getattr(to_reload, k, None) is None:
            raise Exception('Attribute "%s" not found in the model to reload' % k)
        if getattr(model, k) != getattr(to_reload, k):
            raise Exception('Attribute "%s" differs between the current model (%s) '
                            'and the one to reload (%s)'
                            % (k, str(getattr(model, k)), str(getattr(to_reload, k))))

    # copy saved parameters
    for k in model.state_dict().keys():
        if model.state_dict()[k].size() != to_reload.state_dict()[k].size():
            raise Exception("Expected tensor {} of size {}, but got {}".format(
                k, model.state_dict()[k].size(),
                to_reload.state_dict()[k].size()
            ))
        model.state_dict()[k].copy_(to_reload.state_dict()[k])


def print_accuracies(values):
    """
    Pretty plot of accuracies.
    """
    assert all(len(x) == 2 for x in values)
    for name, value in values:
        logger.info('{:<20}: {:>6}'.format(name, '%.3f%%' % (100 * value)))
    logger.info('')


def get_lambda(l, params):
    """
    Compute discriminators' lambdas.
    """
    s = params.lambda_schedule
    if s == 0:
        return l
    else:
        return l * float(min(params.n_total_iter, s)) / s
