from absl import logging

import tensorflow as tf
from tensorflow.keras.losses import (SparseCategoricalCrossentropy)


def getLossFn(cfg, name='loss_function'):
    """Loss function"""
    logging.info(f'Configuring {name}...')

    cfg_loss_fn = cfg['loss_fn']
    
    # get loss function
    if cfg_loss_fn == 'SparseCategoricalCrossentropy':
        cfg_from_logits = cfg['loss_fn_config']['from_logits']
        cfg_reduction = cfg['loss_fn_config']['reduction']
        loss_fn = SparseCategoricalCrossentropy(from_logits=cfg_from_logits, 
                                                reduction=cfg_reduction,
                                                name=name)    
    else:
        raise ValueError('Unknown \'{}\' for being loss function'.format(cfg_loss_fn))
    
    return loss_fn