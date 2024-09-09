import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, GlobalMaxPool2D, GlobalAveragePooling2D, Dense)


def genAttnModule(numChannel_InputFeature, attn_type, attn_setting, name=None):
    if attn_type == 'ChannelAttention0':
        # attn_setting of 'ChannelAttention0' -> list [idx 0:reduction_ratio, 1:activation_dense1, 2:activation_dense2]
        r_ratio = attn_setting[0]
        act_d1 = attn_setting[1]
        act_d2 = attn_setting[2]
        attn_block = ChannelAttention0(channel_output=numChannel_InputFeature,
                                       reduction_ratio=r_ratio, 
                                       activation_dense1=act_d1, 
                                       activation_dense2=act_d2,
                                       name=name)
    elif attn_type == 'SpatialAttention0':
        # attn_setting of 'SpatialAttention0' -> list [idx 0:filter, 1:kernel_size, 2:stride, 3:activation, 4:epsilon]
        num_filter = attn_setting[0]
        kernel_size = attn_setting[1]
        stride = attn_setting[2]
        activation = attn_setting[3]
        epsilon = attn_setting[4]
        attn_block = SpatialAttention0(filters=num_filter,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       activation=activation,
                                       epsilon=epsilon,
                                       name=name)
    else:
        raise ValueError('Unknown attention type \'{}\''.format(attn_type))
    
    return attn_block
 

class ChannelAttention0(tf.keras.layers.Layer):
    """Channel Attenton block from paper 'Squeeze-and-Excitation (SE) Networks [2017]'."""
    def __init__(self, channel_output, reduction_ratio=16, activation_dense1='relu', activation_dense2='sigmoid', name=None, **kwargs):
        super(ChannelAttention0, self).__init__(name=name, **kwargs)
        self.ch_out = channel_output
        self.squeeze = GlobalAveragePooling2D()
        self.dense1 = Dense(units=channel_output // reduction_ratio,
                            activation=activation_dense1,
                            name=name + "_dense1")
        self.dense2 = Dense(units=channel_output,
                            activation=activation_dense2,
                            name=name + "_dense2")
    
    def call(self, inputs, training=False):
        attn = self.squeeze(inputs)
        attn = self.dense1(attn)
        attn = self.dense2(attn)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(inputs, attn)


class SpatialAttention0(tf.keras.layers.Layer):
    """Spatial Attenton block from paper 'Learning for retinal image quality assessment with label regularization [2022]'."""
    def __init__(self, filters=1, kernel_size=1, stride=1, activation='softplus', epsilon=0.1, name=None, **kwargs):
        super(SpatialAttention0, self).__init__(name=name, **kwargs)
        self.conv = Conv2D(filters=filters,
                           kernel_size=kernel_size,
                           strides=stride,
                           padding="SAME",
                           activation=activation,
                           name=name + "_conv")
        self.epsilon = epsilon
        self.globalMaxPool = GlobalMaxPool2D(keepdims=True)

    def call(self, inputs, training=False):
        attn = self.conv(inputs)
        attn = tf.math.add(attn, self.epsilon)
        max = self.globalMaxPool(attn)
        attn = tf.math.divide(attn, max)
        return tf.math.multiply(inputs, attn)
