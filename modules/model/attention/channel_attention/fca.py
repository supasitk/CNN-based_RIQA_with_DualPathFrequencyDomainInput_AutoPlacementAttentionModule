import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense)

from modules.initializers import *


def get_freq_indices(method, num_freq):
    assert method in ['TriangleLow']
    assert num_freq in [1, 2, 4, 8, 16, 32]

    if method == 'TriangleLow':
        all_TriangleLow_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_TriangleLow_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_TriangleLow_indices_x[:num_freq]
        mapper_y = all_TriangleLow_indices_y[:num_freq]
    else:
        raise NotImplementedError
    
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(tf.keras.layers.Layer):
    """Generate dct filters"""
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        
        # fixed DCT init
        self.weight = tf.Variable(initial_value=self.get_dct_filter(height, width, mapper_x, mapper_y, channel),
                                  trainable=False)

        # fixed random init
        #self.weight = tf.Variable(initial_value=tf.random.uniform(shape=[height, width, channel]),
        #                          trainable=False)

        # learnable DCT init
        #self.weight = tf.Variable(initial_value=self.get_dct_filter(height, width, mapper_x, mapper_y, channel),
        #                          trainable=True)

        # learnable random init
        #self.weight = tf.Variable(initial_value=tf.random.uniform(shape=[height, width, channel]),
        #                          trainable=True)


    def call(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, h, w, c = x.shape
        
        x = x * self.weight
        
        result = tf.math.reduce_sum(x, axis=[1,2])
        return result


    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)


    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = np.zeros(shape=[tile_size_x, tile_size_y, channel], dtype=np.float32)
        
        c_part = channel // len(mapper_x)
        
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[t_x, t_y, i * c_part: (i+1)*c_part] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        
        return dct_filter


class ChannelAttention2(tf.keras.layers.Layer):
    """Channel Attenton block from paper 'FcaNet: Frequency Channel Attention Networks [2021]'."""
    def __init__(self, freq_select_method, num_freq, reduction_ratio=16, activation_dense1='relu', activation_dense2='sigmoid', use_bias=True, seed=None, name=None, **kwargs):
        super(ChannelAttention2, self).__init__(name=name, **kwargs)
        self.freq_select_method = freq_select_method
        self.num_freq = num_freq
        self.reduction_ratio = reduction_ratio
        self.activation_dense1 = activation_dense1
        self.activation_dense2 = activation_dense2
        self.use_bias = use_bias
        self.seed = seed
        self.name = name


    def build(self, input_shape):
        input_h = input_shape[1]
        input_w = input_shape[2]
        input_c = input_shape[3]
        self.ch_out = input_c

        mapper_x, mapper_y = get_freq_indices(self.freq_select_method, self.num_freq)
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        mapper_x = [temp_x * (input_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (input_w // 7) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(input_h, input_w, mapper_x, mapper_y, input_c)

        self.dense1 = Dense(units=input_c // self.reduction_ratio,
                            activation=self.activation_dense1,
                            use_bias=self.use_bias,
                            kernel_initializer=DENSE_KERNEL_INIT(seed=self.seed),
                            name=self.name + "_dense1")
        
        self.dense2 = Dense(units=input_c,
                            activation=self.activation_dense2,
                            use_bias=self.use_bias,
                            kernel_initializer=DENSE_KERNEL_INIT(seed=self.seed),
                            name=self.name + "_dense2")
        
    def call(self, x):
        attn = self.dct_layer(x)
        attn = self.dense1(attn)
        attn = self.dense2(attn)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(x, attn)