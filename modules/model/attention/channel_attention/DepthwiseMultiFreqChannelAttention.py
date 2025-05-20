import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Activation, GlobalAveragePooling2D, GlobalMaxPool2D)

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
                    # Freq component#    0       4       8      12      16      20      24      28      32      36      40      44      48
            all_TriangleLow_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4,5,6,6,2,3,5,5,4,4,3,6,5,4,6,5,6,6]
            all_TriangleLow_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3,2,1,2,6,5,3,4,5,4,6,3,5,6,4,6,5,6]
            
            mapper_x = all_TriangleLow_indices_x[:num_freq]
            mapper_y = all_TriangleLow_indices_y[:num_freq]
        else:
            raise NotImplementedError
        
        return mapper_x, mapper_y


class MLP(tf.keras.layers.Layer):
    """Multi-Layer Perceptron"""
    def __init__(self, input_shape_c, reduction_ratio, use_bias, activation_dense1, seed=None, name=None, **kwargs):
        super(MLP, self).__init__(name=name, **kwargs)

        self.mlp = tf.keras.Sequential()
        self.mlp.add(Dense(units=input_shape_c // reduction_ratio,
                           activation=activation_dense1,
                           use_bias=use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                           name=f'{name}_dense1'))
        self.mlp.add(Dense(units=input_shape_c,
                           activation=None,
                           use_bias=use_bias,
                           kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                           name=f'{name}_dense2'))


    def call(self, x):
        out = self.mlp(x)

        return out


class WeightedMLPoutputs(tf.keras.layers.Layer):
    def __init__(self, num_output, seed=None, name=None, **kwargs):
        super(WeightedMLPoutputs, self).__init__(name=name, **kwargs)

        self.W = self.add_weight(shape=[num_output],
                                 initializer=tf.keras.initializers.GlorotUniform(seed=seed), 
                                 dtype=tf.float32, 
                                 trainable=True)


    def call(self, inputs):
        out = list()
        for input, w in zip(inputs, self.W):
            out.append(tf.math.multiply(input, w))
        
        return out


class ChannelAttention3(tf.keras.layers.Layer):
    """
        DepthwiseMultiFreqChannelAttention with
            DepthwiseGlobalSpatialPooling: (null [no using DepthwiseGlobalSpatialPooling] or method ['avg', 'max', 'avg_max']) -> 
            Depthwise2DdctSpatialPooling: (freq_select_method ['TriangleLow'], num_freq [integer]) ->
            DepthwiseMultiFreqReduction: (num_group_of_freq [integer], num_freq_per_group [integer], method ['reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_mean_max']) ->
            MLP: (mlp_arrangement ['share', 'seperate'], mlp_reduction_ratio (integer), mlp_use_bias (Boolean), mlp_activation_dense1 (ActivationName), mlp_weighted_dense2_outputs (Boolean), mlp_activation_dense2 (ActivationName))
    """
    def __init__(self, globalSpaPool_reduction_method, 
                 dctSpaPool_freq_select_method, dctSpaPool_num_freq, 
                 multiFreqReduction_num_group_of_freq, multiFreqReduction_num_freq_per_group, multiFreqReduction_method='reduce_mean',
                 mlp_arrangement='share', mlp_reduction_ratio=16, mlp_use_bias=True, mlp_activation_dense1='relu', mlp_weighted_dense2_outputs=False, mlp_activation_dense2='sigmoid', 
                 seed=None, name=None, **kwargs):
        super(ChannelAttention3, self).__init__(name=name, **kwargs)
        assert globalSpaPool_reduction_method in [None, 'avg', 'max', 'avg_max']
        self.globalSpaPool_reduction_method = globalSpaPool_reduction_method
        assert dctSpaPool_freq_select_method in ['TriangleLow']
        self.dctSpaPool_freq_select_method = dctSpaPool_freq_select_method
        assert multiFreqReduction_num_freq_per_group == (dctSpaPool_num_freq // multiFreqReduction_num_group_of_freq)
        self.dctSpaPool_num_freq = dctSpaPool_num_freq
        self.multiFreqReduction_num_group_of_freq = multiFreqReduction_num_group_of_freq
        self.multiFreqReduction_num_freq_per_group = multiFreqReduction_num_freq_per_group
        assert multiFreqReduction_method in ['reduce_mean', 'reduce_sum', 'reduce_max', 'reduce_mean_max']
        self.multiFreqReduction_method = multiFreqReduction_method
        assert mlp_arrangement in ['share', 'seperate']
        self.mlp_arrangement = mlp_arrangement
        self.mlp_reduction_ratio = mlp_reduction_ratio
        self.mlp_use_bias = mlp_use_bias
        self.mlp_activation_dense1 = mlp_activation_dense1
        self.mlp_weighted_dense2_outputs = mlp_weighted_dense2_outputs
        self.mlp_activation_dense2 = mlp_activation_dense2
        self.seed = seed
        self.name = name
    

    def build(self, input_shape):
        input_h = input_shape[1]
        input_w = input_shape[2]
        input_c = input_shape[3]
        self.ch_out = input_c

        # Depthwise Spatial Pooling includes 
        #   1. DepthwiseGlobalSpatialPooling [if any] and 
        #   2. Depthwise2DdctSpatialPooling
        #-------------------------------------------------------------------------
        #-- 1. DepthwiseGlobalSpatialPooling [if any]
        self.depthwise_global_spatial_pooling_ops = list()
        if self.globalSpaPool_reduction_method is not None:
            if self.globalSpaPool_reduction_method == 'avg':
                self.depthwise_global_spatial_pooling_ops.append(GlobalAveragePooling2D())
            elif self.globalSpaPool_reduction_method == 'max':
                self.depthwise_global_spatial_pooling_ops.append(GlobalMaxPool2D())
            elif self.globalSpaPool_reduction_method == 'avg_max':
                self.depthwise_global_spatial_pooling_ops.append(GlobalAveragePooling2D())
                self.depthwise_global_spatial_pooling_ops.append(GlobalMaxPool2D())
            else:
                raise NotImplementedError

        #-- 2. Depthwise2DdctSpatialPooling
        self.depthwise_2dDCT_spatial_pooling = Depthwise2DdctSpatialPooling(input_h=input_h,
                                                                            input_w=input_w,
                                                                            input_c=input_c,
                                                                            freq_select_method=self.dctSpaPool_freq_select_method,
                                                                            num_freq=self.dctSpaPool_num_freq)
   
        #---- DepthwiseMultiFreqReduction (for 2. Depthwise2DdctSpatialPooling)
        self.depthwise_multiFreq_reduction_methods = list()
        if self.multiFreqReduction_method == 'reduce_mean':
            self.depthwise_multiFreq_reduction_methods.append(tf.math.reduce_mean)
        elif self.multiFreqReduction_method == 'reduce_sum':
            self.depthwise_multiFreq_reduction_methods.append(tf.math.reduce_sum)
        elif self.multiFreqReduction_method == 'reduce_max':
            self.depthwise_multiFreq_reduction_methods.append(tf.math.reduce_max)
        elif self.multiFreqReduction_method == 'reduce_mean_max':
            self.depthwise_multiFreq_reduction_methods.append(tf.math.reduce_mean)
            self.depthwise_multiFreq_reduction_methods.append(tf.math.reduce_max)
        else:
            raise NotImplementedError 

        # MLP
        if self.mlp_arrangement == 'share':
            self.mlp_shared = MLP(input_shape_c=input_c,
                                  reduction_ratio=self.mlp_reduction_ratio,
                                  use_bias=self.mlp_use_bias,
                                  activation_dense1=self.mlp_activation_dense1,
                                  seed=self.seed,
                                  name=f'{self.name}_mlp_shared')
        elif self.mlp_arrangement == 'seperate':
            # for seperated MLP(s), there are 2 set of MLP(s) including 
            #   1. only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
            #   2. MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction] (the number of MLP(s) depends on the number of group(s) of frequency(s)) 
            #-------------------------------------------------------------------------
            #-- 1. only one MLP for output of DepthwiseGlobalSpatialPooling [if any] 
            if self.globalSpaPool_reduction_method is not None:
                self.mlp_seperated_for_depthwiseGlobalSpatialPooling = MLP(input_shape_c=input_c,
                                                                           reduction_ratio=self.mlp_reduction_ratio,
                                                                           use_bias=self.mlp_use_bias,
                                                                           activation_dense1=self.mlp_activation_dense1,
                                                                           seed=self.seed,
                                                                           name=f'{self.name}_mlp_seperated_for_depthwiseGlobalSpatialPooling')
             
            #-- 2. MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction] 
            self.num_mlp_for_depthwise2dDCTspatialPooling = self.multiFreqReduction_num_group_of_freq
            self.mlp_seperated_for_depthwise2dDCTspatialPooling = list()
            for i in range(self.num_mlp_for_depthwise2dDCTspatialPooling):
                self.mlp_seperated_for_depthwise2dDCTspatialPooling.append(MLP(input_shape_c=input_c,
                                                                               reduction_ratio=self.mlp_reduction_ratio,
                                                                               use_bias=self.mlp_use_bias,
                                                                               activation_dense1=self.mlp_activation_dense1,
                                                                               seed=self.seed,
                                                                               name=f'{self.name}_mlp_separated_for_depthwise2dDCTspatialPooling_{i}'))
        else:
            raise NotImplementedError
        
        # Weighting output(s) from (dense2 of) MLP(s) includes
        #   1. weight(s) for output(s) of only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
        #   2. weight(s) for output(s) of MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]
        #-------------------------------------------------------------------------
        if self.mlp_weighted_dense2_outputs:
            #-- 1. weight(s) for output(s) of only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
            self.weighted_mlp_outputs_for_depthwiseGlobalSpatialPooling = list()
            if self.globalSpaPool_reduction_method is not None:
                for _ in range(len(self.depthwise_global_spatial_pooling_ops)):
                    self.weighted_mlp_outputs_for_depthwiseGlobalSpatialPooling.append(WeightedMLPoutputs(num_output=1,
                                                                                                          seed=self.seed,
                                                                                                          name=f'{self.name}_weighted_mlp_outputs_for_depthwiseGlobalSpatialPooling'))

            #-- 2. weight(s) for output(s) of MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]  
            self.weighted_mlp_outputs_for_depthwise2dDCTspatialPooling = list()
            for _ in range(len(self.depthwise_multiFreq_reduction_methods)):
                self.weighted_mlp_outputs_for_depthwise2dDCTspatialPooling.append(WeightedMLPoutputs(num_output=self.multiFreqReduction_num_group_of_freq,
                                                                                                     seed=self.seed,
                                                                                                     name=f'{self.name}_weighted_mlp_outputs_for_depthwise2dDCTspatialPooling'))

        # Activation after tf.math.add_n of output(s) from MLP(s)
        self.activation_dense2_of_mlp = Activation(activation=self.mlp_activation_dense2,
                                                   name=f'{self.name}_mlp_act_dense2')


    def call(self, x):
        # Depthwise Spatial Pooling includes 
        #   1. DepthwiseGlobalSpatialPooling [if any] and 
        #   2. Depthwise2DdctSpatialPooling
        #-------------------------------------------------------------------------
        #-- 1. DepthwiseGlobalSpatialPooling [if any]
        #--   input shape = [B, H, W, C]
        #--   if NOT using DepthwiseGlobalSpatialPooling, output is empty list; 
        #--   else, output is list of Ops elements with tensor shape [B, C] of each one (Ops is number of globalSpaPool_reduction_method)
        out_depthwise_global_spatial_pooling_ops = list()
        if self.globalSpaPool_reduction_method is not None:
            for op in self.depthwise_global_spatial_pooling_ops:
                out_depthwise_global_spatial_pooling_ops.append(op(x))

        #-- 2. Depthwise2DdctSpatialPooling 
        #--   input shape = [B, H, W, C]
        #--   output is list of F elements (freqs) with tensor shape [B, C] of each one (F is number of frequency used)
        out_depthwise_2dDCT_spatial_pooling_for_all_freq = self.depthwise_2dDCT_spatial_pooling(x)

        #---- DepthwiseMultiFreqReduction (for 2. Depthwise2DdctSpatialPooling)
        #----   input is list of F elements (freqs) with tensor shape [B, C] of each one
        #----   output created via 3 step: 
        #----       1st step (grouping): outcome is list of G inner-lists (groups) that each inner-list (group) has (F / G) elements (num_freq per group) with tensor shape [B, C] of each one (G is number of group of frequency)
        #----       2nd step (stacking): outcome is list of G stacked tensors with shape [(F/G), B, C] of each one
        #----       3rd step (reduction): outcome is list of R inner-lists (reduction_methods) with G reduced tensors of each one with shape [B, C] of each one (R is number of reduction methods)
        #------ 1st step
        assert len(out_depthwise_2dDCT_spatial_pooling_for_all_freq) == self.dctSpaPool_num_freq
        grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq = [out_depthwise_2dDCT_spatial_pooling_for_all_freq[start_idx_grp:start_idx_grp+self.multiFreqReduction_num_freq_per_group] for start_idx_grp in range(0, len(out_depthwise_2dDCT_spatial_pooling_for_all_freq), self.multiFreqReduction_num_freq_per_group)]
        #------ 2nd step
        assert len(grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq) == self.multiFreqReduction_num_group_of_freq
        stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq = [tf.stack(freq_grp, axis=0) for freq_grp in grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq]
        #------ 3rd step
        assert len(stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq) == self.multiFreqReduction_num_group_of_freq
        all_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq = list()
        for reduction_method in self.depthwise_multiFreq_reduction_methods:
            all_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq.append([reduction_method(freq_grp, axis=0) for freq_grp in stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq])
        assert len(all_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq) == len(self.depthwise_multiFreq_reduction_methods)


        # MLP has 2 parts:
        #   1. only one MLP for output of DepthwiseGlobalSpatialPooling [if any] and
        #   2. MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]
        #-------------------------------------------------------------------------
        #-- 1. only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
        all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling = list()
        if self.globalSpaPool_reduction_method is not None:
            assert len(out_depthwise_global_spatial_pooling_ops) == len(self.depthwise_global_spatial_pooling_ops)
            for out_op in out_depthwise_global_spatial_pooling_ops:
                ch_map_outs = list()
                if self.mlp_arrangement == 'share':
                    ch_map_outs.append(self.mlp_shared(out_op))
                elif self.mlp_arrangement == 'seperate':
                    ch_map_outs.append(self.mlp_seperated_for_depthwiseGlobalSpatialPooling(out_op))
                all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling.append(ch_map_outs)

        #-- 2. MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]
        all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling = list()
        for reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq in all_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq:
            assert len(reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq) == self.multiFreqReduction_num_group_of_freq
            channel_map_outs = list()
            if self.mlp_arrangement == 'share':
                for this_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling in reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq:
                    channel_map_outs.append(self.mlp_shared(this_reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling))
            elif self.mlp_arrangement == 'seperate':
                assert len(reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq) == len(self.mlp_seperated_for_depthwise2dDCTspatialPooling)
                for input, mlp in zip(reduced_stacked_grouped_out_depthwise_2dDCT_spatial_pooling_for_all_freq, self.mlp_seperated_for_depthwise2dDCTspatialPooling):
                    channel_map_outs.append(mlp(input))
            all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling.append(channel_map_outs)


        # Weighting output(s) from (dense2 of) MLP(s)
        #   1. weight(s) for output(s) of only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
        #   2. weight(s) for output(s) of MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]
        #-------------------------------------------------------------------------
        #-- 1. weight(s) for output(s) of only one MLP for output of DepthwiseGlobalSpatialPooling [if any]
        weighted_all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling = list()
        if self.globalSpaPool_reduction_method is not None:
            if self.mlp_weighted_dense2_outputs:
                assert len(all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling) == len(self.weighted_mlp_outputs_for_depthwiseGlobalSpatialPooling)
                for out, weight in zip(all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling, self.weighted_mlp_outputs_for_depthwiseGlobalSpatialPooling):
                    weighted_all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling.append(weight(out))
            else:
                weighted_all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling = all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling

        #-- 2. weight(s) for output(s) of MLP(s) for output of Depthwise2DdctSpatialPooling [with DepthwiseMultiFreqReduction]
        weighted_all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling = list()
        if self.mlp_weighted_dense2_outputs:
            assert len(all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling) == len(self.weighted_mlp_outputs_for_depthwise2dDCTspatialPooling)
            for outs, weights in zip(all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling, self.weighted_mlp_outputs_for_depthwise2dDCTspatialPooling):
                weighted_all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling.append(weights(outs))
        else:
            weighted_all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling = all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling


        # Merging channel map output(s) from executing DepthwiseGlobalSpatialPooling and Depthwise2DdctSpatialPooling
        merged_weighted_all_reduced_channel_map_outs = list()
        if self.globalSpaPool_reduction_method is not None:
            for weighted_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling in weighted_all_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling:
                merged_weighted_all_reduced_channel_map_outs.extend(weighted_reduced_channel_map_outs_of_depthwiseGlobalSpatialPooling)
        for weighted_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling in weighted_all_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling:
            merged_weighted_all_reduced_channel_map_outs.extend(weighted_reduced_channel_map_outs_of_depthwise2dDCTspatialPooling)


        # Activation after tf.math.add_n of output(s) from MLP(s)
        channel_map_outs_sum = tf.math.add_n(merged_weighted_all_reduced_channel_map_outs)
        attn = self.activation_dense2_of_mlp(channel_map_outs_sum)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])

        out = tf.math.multiply(x, attn)

        return out