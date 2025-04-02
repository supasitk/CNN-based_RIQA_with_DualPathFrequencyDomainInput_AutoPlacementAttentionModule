import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Activation)

from modules.initializers import *


class oneFreq2DdctSpatialPooling(tf.keras.layers.Layer):
    """Generate dct filters"""
    def __init__(self, height, width, freq_x, freq_y, channel):
        super(oneFreq2DdctSpatialPooling, self).__init__()
        
        # fixed DCT init
        self.weight = tf.Variable(initial_value=self._get_dct_filter(height, width, freq_x, freq_y, channel),
                                  trainable=False)


    def call(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, h, w, c = x.shape
        
        x = x * self.weight
        
        result = tf.math.reduce_sum(x, axis=[1,2])
        return result


    def _build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    
    def _get_dct_filter(self, tile_size_x, tile_size_y, freq_x, freq_y, channel):

        dct_filter = np.zeros(shape=[tile_size_x, tile_size_y, channel], dtype=np.float32)
        
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[t_x, t_y, 0:channel] = self._build_filter(t_x, freq_x, tile_size_x) * self._build_filter(t_y, freq_y, tile_size_y)
        
        return dct_filter


class Depthwise2DdctSpatialPooling(tf.keras.layers.Layer):
    """Depthwise2DdctSpatialPooling"""
    def __init__(self, input_h, input_w, input_c, freq_select_method, num_freq):
        super(Depthwise2DdctSpatialPooling, self).__init__()

        mapper_x, mapper_y = self._get_freq_indices(freq_select_method, num_freq)
        assert len(mapper_x) == len(mapper_y)
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        mapper_x = [temp_x * (input_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (input_w // 7) for temp_y in mapper_y]

        self.depthwise_2dDCT_spatial_pooling = list()
        for freq_x, freq_y in zip(mapper_x, mapper_y):
            self.depthwise_2dDCT_spatial_pooling.append(oneFreq2DdctSpatialPooling(height=input_h,
                                                                                   width=input_w,
                                                                                   freq_x=freq_x,
                                                                                   freq_y=freq_y,
                                                                                   channel=input_c))


    def call(self, x):
        out_depthwise_2dDCT_spatial_pooling = list()
        for DCT2d_spatial_pooling_with_this_freq in self.depthwise_2dDCT_spatial_pooling:
            out_depthwise_2dDCT_spatial_pooling.append(DCT2d_spatial_pooling_with_this_freq(x))

        return out_depthwise_2dDCT_spatial_pooling


    def _get_freq_indices(self, method, num_freq):
        assert method in ['TriangleLow']
        
        if method == 'TriangleLow':
            all_TriangleLow_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
            all_TriangleLow_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
            
            mapper_x = all_TriangleLow_indices_x[:num_freq]
            mapper_y = all_TriangleLow_indices_y[:num_freq]
        else:
            raise NotImplementedError
        
        return mapper_x, mapper_y


class ChannelAttention3_1(tf.keras.layers.Layer):
    """
        DepthwiseMultiFreqChannelAttention with
            Depthwise2DdctSpatialPooling: True ->
            DepthwiseMultiFreqReduction: False ->
            MLP: Shared
    """
    def __init__(self, dctSpaPool_freq_select_method, dctSpaPool_num_freq, mlp_reduction_ratio=16, mlp_use_bias=True, 
                 mlp_activation_dense1='relu', mlp_activation_dense2='sigmoid', seed=None, name=None, **kwargs):
        super(ChannelAttention3_1, self).__init__(name=name, **kwargs)
        self.dctSpaPool_freq_select_method = dctSpaPool_freq_select_method
        self.dctSpaPool_num_freq = dctSpaPool_num_freq
        self.mlp_reduction_ratio = mlp_reduction_ratio
        self.mlp_use_bias = mlp_use_bias
        self.mlp_activation_dense1 = mlp_activation_dense1
        self.mlp_activation_dense2 = mlp_activation_dense2
        self.seed = seed
        self.name = name
    

    def build(self, input_shape):
        input_h = input_shape[1]
        input_w = input_shape[2]
        input_c = input_shape[3]
        self.ch_out = input_c

        # Depthwise2DdctSpatialPooling
        self.depthwise_2dDCT_spatial_pooling = Depthwise2DdctSpatialPooling(input_h=input_h,
                                                                            input_w=input_w,
                                                                            input_c=input_c,
                                                                            freq_select_method=self.dctSpaPool_freq_select_method,
                                                                            num_freq=self.dctSpaPool_num_freq)
    
        # MLP (shared)
        self.mlp = tf.keras.Sequential()
        self.mlp.add(Dense(units=input_c // self.mlp_reduction_ratio,
                           activation=self.mlp_activation_dense1,
                           use_bias=self.mlp_use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=self.seed),
                           name=f'{self.name}_mlp_dense1'))
        self.mlp.add(Dense(units=input_c,
                           activation=None,
                           use_bias=self.mlp_use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=self.seed),
                           name=f'{self.name}_mlp_dense2'))

        self.activation_dense2_of_mlp = Activation(activation=self.mlp_activation_dense2,
                                                   name=f'{self.name}_mlp_act_dense2')


    def call(self, x):
        out_depthwise_2dDCT_spatial_pooling_for_all_freq = self.depthwise_2dDCT_spatial_pooling(x)  

        channel_map_outs = list()
        for out_depthwise_2dDCT_spatial_pooling_for_this_freq in out_depthwise_2dDCT_spatial_pooling_for_all_freq:
            channel_map_outs.append(self.mlp(out_depthwise_2dDCT_spatial_pooling_for_this_freq))
        
        channel_map_outs_sum = tf.math.add_n(channel_map_outs)

        attn = self.activation_dense2_of_mlp(channel_map_outs_sum)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])

        out = tf.math.multiply(x, attn)

        return out