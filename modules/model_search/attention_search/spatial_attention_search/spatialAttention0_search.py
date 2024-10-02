import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, GlobalMaxPool2D)


class SpatialAttention0_Search(tf.keras.layers.Layer):
    """Spatial Attenton block from paper 'Learning for retinal image quality assessment with label regularization [2022]'."""
    def __init__(self, block_id, kernel_size_search_infos, filters=1, stride=1, activation='softplus', epsilon=0.1, name=None, **kwargs):
        super(SpatialAttention0_Search, self).__init__(name=name, **kwargs)
        ### TEST ###
        print('block {}: kernel_size_search_infos: {}'.format(block_id, kernel_size_search_infos))
        kernel_size = 1
        ### TEST ###
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding="SAME",
                           activation=activation,
                           name=name + "_conv")
        self.epsilon = epsilon
        self.globalMaxPool = GlobalMaxPool2D(keepdims=True)

    def call(self, inputs, alphas_intra_attn_weights, training=False):
        attn = self.conv(inputs)
        attn = tf.math.add(attn, self.epsilon)
        max = self.globalMaxPool(attn)
        attn = tf.math.divide(attn, max)
        return tf.math.multiply(inputs, attn)
