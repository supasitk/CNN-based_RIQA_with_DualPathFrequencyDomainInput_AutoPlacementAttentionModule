import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, GlobalMaxPool2D)


class BranchInput(tf.keras.layers.Layer):
    """Gather input for each branch of model's structure according to specific indices of input channel"""
    def __init__(self, Branch_idx_input_ch, axis=-1, name=None, **kwargs):
        super(BranchInput, self).__init__(name=name, **kwargs)
        self.Branch_idx_input_ch = Branch_idx_input_ch
        self.axis = axis
    
    def call(self, inputs, training=False):
        return tf.gather(inputs, self.Branch_idx_input_ch, axis=self.axis)
    

class SpatialAttention0(tf.keras.layers.Layer):
    """Spatial Attenton block from paper 'Learning for retinal image quality assessment with label regularization [2022]'."""
    def __init__(self, filters=1, kernel_size=1, stride=1, activation='softplus', epsilon=0.1, name=None, **kwargs):
        super(SpatialAttention0, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding="SAME",
                           activation=activation,
                           name=name)
        self.epsilon = epsilon
        self.globalMaxPool = GlobalMaxPool2D(keepdims=True)

    def call(self, inputs, training=False):
        attn = self.conv(inputs)
        attn = tf.math.add(attn, self.epsilon)
        max = self.globalMaxPool(attn)
        attn = tf.math.divide(attn, max)
        return tf.math.multiply(inputs, attn)