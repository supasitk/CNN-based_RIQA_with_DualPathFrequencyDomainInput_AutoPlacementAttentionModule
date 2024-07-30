from absl import logging

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import (ExponentialDecay, CosineDecay)


def getLRscheduler(lr_schedule, lr_config, total_steps, steps_per_epoch, name='LRscheduler'):
    """Learning rate scheduler"""
    logging.info(f'Configuring {name}...')

    if lr_schedule == 'Fix':
        lr_scheduler = lr_config['lr_init']
    elif lr_schedule == 'ExponentialDecay':
        lr_init = lr_config['lr_init']
        lr_min = lr_config['lr_min']
        decay_unit = lr_config['decay_unit']
        decay_on_every_unit = lr_config['decay_on_every_unit']
        staircase = lr_config['staircase']

        decay_on_every_step = int(decay_on_every_unit) if decay_unit == 'step' else int(decay_on_every_unit * steps_per_epoch)
        num_times_of_decay = total_steps // decay_on_every_step
        decay_rate = (lr_min / lr_init) ** (1 / num_times_of_decay)

        lr_scheduler = ExponentialDecay(initial_learning_rate=lr_init,
                                        decay_steps=decay_on_every_step,
                                        decay_rate=decay_rate,
                                        staircase=staircase)
    elif lr_schedule == 'CosineDecay':
        lr_init = lr_config['lr_init']
        lr_alpha = lr_config['lr_alpha']
        decay_steps = total_steps

        lr_scheduler = CosineDecay(initial_learning_rate=lr_init,
                                   decay_steps=decay_steps,
                                   alpha=lr_alpha)
    else:
        raise ValueError('Unknown \'{}\' for being learning rate scheduler'.format(lr_schedule))
    
    return lr_scheduler
