from absl import logging

import tensorflow as tf
from tensorflow.keras.losses import (SparseCategoricalCrossentropy, CategoricalCrossentropy)


class LossFn(object):
    def __init__(self, cfg, ds_cfg, train_num_img_dict, name='loss_function'):
        """Loss function"""
        logging.info(f'Configuring {name}...')
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        self.train_num_img_dict = train_num_img_dict
        self.name = name
        
        self.loss_fn = self._getLossFn()    


    def _getClassWeight(self):
        """get class weight, 
           optional list mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only).
           These will cause the model to "pay more attention" to examples from an under-represented class. 
           Note, however, that this does not increase in any way the amount of information of your dataset.

           weight for each class #n = (1 / number_of_training_data_for_class_#n) * (total_number_of_training_data / num_classes)
        """
        num_class = self.ds_cfg['class_num']
        total_num_train_data = self.train_num_img_dict['total_num_img']
        
        list_class_weights = list()
        for class_id in range(num_class):
            this_class_num_train_data = self.train_num_img_dict['list_num_img_of_each_class_id'][class_id]
            this_class_weight = (1 / this_class_num_train_data) * (total_num_train_data / num_class)
            list_class_weights.append(this_class_weight)
        
        class_weights = tf.constant(list_class_weights, dtype=tf.dtypes.float32)

        # show class weights
        name_class = self.ds_cfg['class_name']
        str_class_weights = ""
        for class_id in range(num_class):
            str_class_weights += f"{class_id}({name_class[class_id]}): {class_weights[class_id].numpy():.2f}, "
        logging.info(f'\tClass-weighting loss function (during training only) with {str_class_weights}')

        return class_weights


    def _getLossFn(self):
        """get loss function"""
        cfg_loss_fn = self.cfg['loss_fn']
        
        if cfg_loss_fn == 'SparseCategoricalCrossentropy':
            cfg_from_logits = self.cfg['loss_fn_config']['from_logits']
            cfg_reduction = self.cfg['loss_fn_config']['reduction']
            cfg_train_with_class_weight = self.cfg['loss_fn_config']['train_with_class_weight']
            self.class_weight = None
            if cfg_train_with_class_weight:
                self.class_weight = self._getClassWeight()
            loss_fn = SparseCategoricalCrossentropy(from_logits=cfg_from_logits, 
                                                    reduction=cfg_reduction,
                                                    name=self.name)
        elif cfg_loss_fn == 'SparseCategoricalCrossentropyWithLabelSmoothing':
            cfg_from_logits = self.cfg['loss_fn_config']['from_logits']
            cfg_reduction = self.cfg['loss_fn_config']['reduction']
            cfg_train_with_class_weight = self.cfg['loss_fn_config']['train_with_class_weight']
            self.class_weight = None
            if cfg_train_with_class_weight:
                self.class_weight = self._getClassWeight()
            cfg_label_smoothing = self.cfg['loss_fn_config']['label_smoothing']
            loss_fn = CategoricalCrossentropy(from_logits=cfg_from_logits,
                                              reduction=cfg_reduction,
                                              label_smoothing=cfg_label_smoothing,
                                              name=self.name)
        else:
            raise ValueError('Unknown \'{}\' for being loss function'.format(cfg_loss_fn))
        
        return loss_fn
    

    def compute_loss(self, labels, logits, training=False):
        """compute loss value"""
        cfg_loss_fn = self.cfg['loss_fn']

        if cfg_loss_fn == 'SparseCategoricalCrossentropy':
            sample_weight = None
            if training and (self.class_weight is not None):
                sample_weight = tf.gather(self.class_weight, labels)                
            loss_value = self.loss_fn(labels, logits, sample_weight=sample_weight)
        elif cfg_loss_fn == 'SparseCategoricalCrossentropyWithLabelSmoothing':
            sample_weight = None
            if training and (self.class_weight is not None):
                sample_weight = tf.gather(self.class_weight, labels)
            num_class = self.ds_cfg['class_num']
            one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_class)
            loss_value = self.loss_fn(one_hot_labels, logits, sample_weight=sample_weight)

        return loss_value