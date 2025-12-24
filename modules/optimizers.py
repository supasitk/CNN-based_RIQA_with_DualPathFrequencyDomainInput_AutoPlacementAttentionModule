from absl import logging

import tensorflow as tf
from tensorflow.keras.optimizers import (SGD, Adam)

from modules.lr_schedulers import (getLRscheduler)

def getOptimizer(cfg, total_steps, steps_per_epoch, name='optimizer', arch_search=False):
    """Optimizer"""
    # get LR scheduler
    if arch_search == False:
        cfg_lr_schedule = cfg['lr_schedule']
        cfg_lr_config = cfg['lr_config']
        lr_scheduler_name = 'LRscheduler'
    else:
        cfg_lr_schedule = cfg['arch_lr_schedule']
        cfg_lr_config = cfg['arch_lr_config']
        lr_scheduler_name = 'arch_LRscheduler'

    lr_scheduler = getLRscheduler(cfg_lr_schedule, cfg_lr_config, total_steps, steps_per_epoch, name=lr_scheduler_name)

    # get optimizer
    logging.info(f'Configuring {name}...')

    if arch_search == False:
        cfg_optimizer = cfg['optimizer']
        cfg_optimizer_config = cfg['optimizer_config']
        cfg_wd = cfg['weights_decay']
    else:
        cfg_optimizer = cfg['arch_optimizer']
        cfg_optimizer_config = cfg['arch_optimizer_config']
        cfg_wd = cfg['arch_weights_decay']

    # get optimizer
    if cfg_optimizer == 'SGD':
        cfg_momentum = cfg_optimizer_config['momentum']
        cfg_nesterov = cfg_optimizer_config['nesterov']
        optimizer = SGD(learning_rate=lr_scheduler,
                        momentum=cfg_momentum,
                        nesterov=cfg_nesterov,
                        weight_decay=cfg_wd,
                        name=name + '_sgd')
    elif cfg_optimizer == 'Adam':
        cfg_beta_1 = cfg_optimizer_config['beta_1']
        cfg_beta_2 = cfg_optimizer_config['beta_2']
        optimizer = Adam(learning_rate=lr_scheduler,
                         beta_1=cfg_beta_1,
                         beta_2=cfg_beta_2,
                         weight_decay=cfg_wd,
                         name=name + '_adam')
    else:
        raise ValueError('Unknown \'{}\' for being optimizer'.format(cfg_optimizer))
    
    return optimizer