import tensorflow as tf
from tensorflow.keras.layers import (Conv2D)

from modules.initializers import *
from modules.model.layer.CustomLayer import (ChannelPooling)


class SpatialAttention1(tf.keras.layers.Layer):
    """(CBAM_SpatialAttention) Spatial Attenton block from paper 'CBAM: Convolutional Block Attention Module [2018]'."""
    def __init__(self, channel_pooling_methods=['avg', 'max'], filters=1, kernel_size=7, strides=1, activation='sigmoid', use_bias=False, 
                 seed=None, name=None, **kwargs):
        super(SpatialAttention1, self).__init__(name=name, **kwargs)
        
        # get operation(s) of channel pooling
        self.channel_pooling_ops = list()
        for channel_pooling_method in channel_pooling_methods:
            channel_pooling_op = ChannelPooling(method=channel_pooling_method, 
                                                axis=-1,
                                                keepdims=True,
                                                name=f'{name}_ChPool_{channel_pooling_method}')
            self.channel_pooling_ops.append(channel_pooling_op)
        
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding="SAME",
                           activation=activation,
                           kernel_initializer=CONV2D_KERNEL_INIT(seed=seed),
                           use_bias=use_bias,
                           name=f'{name}_conv')

    def call(self, inputs, training=False):
        ch_pool_outs = list()
        for ch_pool_op in self.channel_pooling_ops:
            this_ch_pool_out = ch_pool_op(inputs)
            ch_pool_outs.append(this_ch_pool_out)
        ch_pool_outs_concat = tf.concat(ch_pool_outs, axis=-1)

        attn = self.conv(ch_pool_outs_concat)

        return tf.math.multiply(inputs, attn)