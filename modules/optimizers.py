from absl import logging

import tensorflow as tf
from tensorflow.keras.optimizers import (SGD)

from modules.lr_schedulers import (getLRscheduler)

def getOptimizer(cfg, total_steps, steps_per_epoch, name='optimizer'):
    """Optimizer"""
    # get LR scheduler
    lr_scheduler = getLRscheduler(cfg['lr_schedule'], cfg['lr_config'], total_steps, steps_per_epoch)

    # get optimizer
    logging.info(f'Configuring {name}...')

    cfg_optimizer = cfg['optimizer']
    cfg_wd = cfg['weights_decay']

    # get optimizer
    if cfg_optimizer == 'SGD':
        cfg_momentum = cfg['optimizer_config']['momentum']
        cfg_nesterov = cfg['optimizer_config']['nesterov']
        optimizer = SGD(learning_rate=lr_scheduler,
                        momentum=cfg_momentum,
                        nesterov=cfg_nesterov,
                        weight_decay=cfg_wd,
                        name=name)
    else:
        raise ValueError('Unknown \'{}\' for being optimizer'.format(cfg_optimizer))
    
    return optimizer