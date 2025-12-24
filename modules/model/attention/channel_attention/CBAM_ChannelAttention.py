import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Activation)

from modules.initializers import *


class ChannelAttention1(tf.keras.layers.Layer):
    """(CBAM_ChannelAttention) Channel Attenton block from paper 'CBAM: Convolutional Block Attention Module [2018]'."""
    def __init__(self, channel_output, spatial_pooling_methods=['avg', 'max'], reduction_ratio=16, use_bias=True, activation_dense1='relu', activation_dense2='sigmoid', 
                 seed=None, name=None, **kwargs):
        super(ChannelAttention1, self).__init__(name=name, **kwargs)

        self.ch_out = channel_output

        # get operation(s) of spatial pooling
        self.spatial_pooling_ops = list()
        for spatial_pooling_method in spatial_pooling_methods:
            if spatial_pooling_method == 'avg':
                spatial_pooling_op = GlobalAveragePooling2D(name=f'{name}_spatialpool_avg')
            elif spatial_pooling_method == 'max':
                spatial_pooling_op = GlobalMaxPool2D(name=f'{name}_spatialpool_max')
            else:
                raise ValueError('Unknown method \'{}\' for SpatialPooling layer'.format(spatial_pooling_method))
            self.spatial_pooling_ops.append(spatial_pooling_op)

        self.mlp = tf.keras.Sequential()
        self.mlp.add(Dense(units=channel_output // reduction_ratio,
                           activation=activation_dense1,
                           use_bias=use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                           name=f'{name}_mlp_dense1'))
        self.mlp.add(Dense(units=channel_output,
                           activation=None,
                           use_bias=use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                           name=f'{name}_mlp_dense2'))

        self.activation_dense2 = Activation(activation=activation_dense2, 
                                            name=f'{name}_act')
        
    def call(self, inputs, training=False):
        channel_map_outs = list()
        for spatial_pool_op in self.spatial_pooling_ops:
            this_channel_pool_out = spatial_pool_op(inputs)
            this_channel_pool_out = self.mlp(this_channel_pool_out)
            channel_map_outs.append(this_channel_pool_out)

        channel_map_outs_sum = tf.math.add_n(channel_map_outs)

        attn = self.activation_dense2(channel_map_outs_sum)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])

        return tf.math.multiply(inputs, attn)