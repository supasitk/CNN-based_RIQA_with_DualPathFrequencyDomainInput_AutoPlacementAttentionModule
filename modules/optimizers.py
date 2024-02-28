from absl import logging

import tensorflow as tf
from tensorflow.keras.optimizers import (SGD)

def getOptimizer(cfg, name='optimizer'):
    """Optimizer"""
    logging.info(f'Configuring {name}...')

    cfg_optimizer = cfg['optimizer']
    cfg_lr_init = cfg['lr_init']
#    cfg_wd = cfg['weights_decay']

    # get optimizer
    if cfg_optimizer == 'SGD':
        cfg_momentum = cfg['optimizer_config']['momentum']
        cfg_nesterov = cfg['optimizer_config']['nesterov']
        optimizer = SGD(learning_rate=cfg_lr_init,
                        momentum=cfg_momentum,
                        nesterov=cfg_nesterov,
#                        weight_decay=cfg_wd,
                        name=name)
    else:
        raise ValueError('Unknown \'{}\' for being optimizer'.format(cfg_optimizer))
    
    return optimizer