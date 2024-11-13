from absl import logging

import tensorflow as tf
from tensorflow.keras.losses import (SparseCategoricalCrossentropy, CategoricalCrossentropy)


class LossFn(object):
    def __init__(self, cfg, ds_cfg, name='loss_function'):
        """Loss function"""
        logging.info(f'Configuring {name}...')
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        self.name = name
        
        self.loss_fn = self._getLossFn()    


    def _getLossFn(self):
        """get loss function"""
        cfg_loss_fn = self.cfg['loss_fn']
        
        if cfg_loss_fn == 'SparseCategoricalCrossentropy':
            cfg_from_logits = self.cfg['loss_fn_config']['from_logits']
            cfg_reduction = self.cfg['loss_fn_config']['reduction']
            loss_fn = SparseCategoricalCrossentropy(from_logits=cfg_from_logits, 
                                                    reduction=cfg_reduction,
                                                    name=self.name)
        elif cfg_loss_fn == 'SparseCategoricalCrossentropyWithLabelSmoothing':
            cfg_from_logits = self.cfg['loss_fn_config']['from_logits']
            cfg_reduction = self.cfg['loss_fn_config']['reduction']
            cfg_label_smoothing = self.cfg['loss_fn_config']['label_smoothing']
            loss_fn = CategoricalCrossentropy(from_logits=cfg_from_logits,
                                              reduction=cfg_reduction,
                                              label_smoothing=cfg_label_smoothing,
                                              name=self.name)
        else:
            raise ValueError('Unknown \'{}\' for being loss function'.format(cfg_loss_fn))
        
        return loss_fn
    

    def compute_loss(self, labels, logits):
        """compute loss value"""
        cfg_loss_fn = self.cfg['loss_fn']

        if cfg_loss_fn == 'SparseCategoricalCrossentropy':
            loss_value = self.loss_fn(labels, logits)
        elif cfg_loss_fn == 'SparseCategoricalCrossentropyWithLabelSmoothing':
            num_class = self.ds_cfg['class_num']
            one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_class)
            loss_value = self.loss_fn(one_hot_labels, logits)

        return loss_value