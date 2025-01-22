import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (Softmax)
from modules.dataset.dataset_statistic.compute_mean_std_of_dataset.eyeq import *


class BranchInput(tf.keras.layers.Layer):
    """Gather input for each branch of model's structure according to specific indices of input channel"""
    def __init__(self, Branch_idx_input_ch, axis=-1, name=None, **kwargs):
        super(BranchInput, self).__init__(name=name, **kwargs)
        self.Branch_idx_input_ch = Branch_idx_input_ch
        self.axis = axis
    
    def call(self, inputs, training=False):
        return tf.gather(inputs, self.Branch_idx_input_ch, axis=self.axis)


class SplitSoftmax(tf.keras.layers.Layer):
    """Split Softmax Layer"""
    def __init__(self, num_blocks, num_ops, size_splits, name='SplitSoftmax', **kwargs):
        super(SplitSoftmax, self).__init__(name=name, **kwargs)
        self.num_blocks = num_blocks
        self.num_ops = num_ops
        self.size_splits = size_splits
        self.soft_max_func = Softmax(axis=-1)
    
    def call(self, value, training=False):
        splitsotfmax_values = []
        for block_id in range(self.num_blocks):
            splitsotfmax_values.append(tf.concat([self.soft_max_func(t) for t in tf.split(value[block_id], self.size_splits)], axis=0))
        
        return tf.reshape(splitsotfmax_values, [self.num_blocks, self.num_ops])


class Skip(tf.keras.layers.Layer):
    """Skip Layer"""
    def __init__(self, name='Skip', **kwargs):
        super(Skip, self).__init__(name=name, **kwargs)
    
    def call(self, x, alphas_intra_attn_weights, training=False):
        return x


class ImageResize(tf.keras.layers.Layer):
    """Image resize Layer"""
    def __init__(self, new_height, new_width, name='ImageResize', **kwargs):
        super(ImageResize, self).__init__(name=name, **kwargs)
        self.new_h = new_height
        self.new_w = new_width
    
    def call(self, x, training=False):
        return tf.image.resize(x, [self.new_h, self.new_w])
    

class ImageRGBtoYUV(tf.keras.layers.Layer):
    """Image RGBtoYUV Layer"""
    def __init__(self, name='ImageRGBtoYUV', **kwargs):
        super(ImageRGBtoYUV, self).__init__(name=name, **kwargs)
    
    def call(self, x, training=False):
        return tf.image.rgb_to_yuv(x)

      
class Depthwise_DCT2d(tf.keras.layers.Layer):
    """ Generate layer of 2D-DCT filters for each input channel, 
            - For each input channel, selecting frequency compenents (using each one for each filter) according to 'freq_component_selection_of_each_channel' argument from given 'freq_component_set' 
            - Each filter's size is 'tile_size_x' * 'tile_size_y' 
        and Transforming input (size of [B, H, W, C]) into 2D-DCT coefficients by convolving seperately these 2D-DCT filters for each input channel with strides ([B, H, W, C]) [1, 'tile_size_x', 'tile_size_y', 1], 
        then concatenates the results together,
        the Output layer's size is [B , H / tile_size_x, W / tile_size_y, sum of 'num_selected_freq_component_of_each_channel' argument]
    """
    def __init__(self, tile_size_x=8, tile_size_y=8, freq_component_set=64, freq_component_selection_of_each_channel=None, num_selected_freq_component_of_each_channel=None, normalize=None, name='Depthwise_DCT2d', **kwargs):
        super(Depthwise_DCT2d, self).__init__(name=name, **kwargs)
        self.tile_size_x = tile_size_x 
        self.tile_size_y = tile_size_y
        self.freq_component_set = freq_component_set
        self.freq_component_selection_of_each_channel = freq_component_selection_of_each_channel
        self.num_selected_freq_component_of_each_channel = num_selected_freq_component_of_each_channel
        self.normalize = normalize 
        self.name = name
    

    def build(self, input_shape):
        self.in_channel = input_shape[-1]
        
        # get freq_idxs_u and freq_idxs_v of selected frequency components for each channel (dict with key(ch_num)-value(list of freq_idxs_u/v) returned)
        freq_idxs_u_for_channels, freq_idxs_v_for_channels = self._get_freq_indices(self.freq_component_set, self.freq_component_selection_of_each_channel, self.num_selected_freq_component_of_each_channel)
        assert (len(freq_idxs_u_for_channels) == self.in_channel) and (len(freq_idxs_v_for_channels) == self.in_channel)

        # fixed DCT filters init
        self.weight = list() # list of 2D-DCT filters set for each corresponding input channel (index of list correspond to index of input channel)
        for ch_num in range(self.in_channel):
            self.weight.append(tf.Variable(initial_value=self._get_dct_filter(self.tile_size_x, self.tile_size_y, freq_idxs_u_for_channels[ch_num], freq_idxs_v_for_channels[ch_num]),
                                           trainable=False))

        # DCT normalization
        if self.normalize is not None:
            idxs_of_stats_for_normalize = list()
            for ch_num in range(self.in_channel):
                this_ch_idxs_of_stats_for_normalize = [(ch_num * self.freq_component_set) + f for f in self.selected_freq_for_channels[ch_num]]
                idxs_of_stats_for_normalize.extend(this_ch_idxs_of_stats_for_normalize)
            
            if self.normalize == 'EyeQpreprocessedTrain_imgRGBresize448rescaleMinusOneToOneTransformToYUVto8x8DCT192freq':
                norm_stats_mean = [EyeQpreprocessedTrain_imgRGBresize448rescaleMinusOneToOneTransformToYUVto8x8DCT192freq['mean'][i] for i in idxs_of_stats_for_normalize]
                norm_stats_var = [EyeQpreprocessedTrain_imgRGBresize448rescaleMinusOneToOneTransformToYUVto8x8DCT192freq['var'][i] for i in idxs_of_stats_for_normalize]
            else:
                raise ValueError

            self.norm_mean = norm_stats_mean
            self.norm_var = norm_stats_var
            self.norm_std = tf.math.sqrt(norm_stats_var)
            
            # ######### TEST #########
            # print('#'*50)
            # print('idxs_of_stats_for_normalize ({}): {}'.format(np.shape(idxs_of_stats_for_normalize), idxs_of_stats_for_normalize))
            # print('-'*50)
            # print('norm_mean ({}): {}'.format(np.shape(self.norm_mean), self.norm_mean))
            # print('-'*50)
            # print('norm_var ({}): {}'.format(np.shape(self.norm_var), self.norm_var))
            # print('-'*50)
            # print('norm_std ({}): {}'.format(np.shape(self.norm_std), self.norm_std))
            # print('#'*50)
            # ######### TEST #########


    def call(self, x):
        out = list()
        for ch_num in range(self.in_channel):
            this_ch_x = tf.gather(x, [ch_num], axis=-1)
            out_this_ch_x = tf.nn.conv2d(input=this_ch_x,
                                         filters=self.weight[ch_num],
                                         strides=[1, self.tile_size_x, self.tile_size_y, 1],
                                         padding='VALID',
                                         data_format='NHWC')
            out.append(out_this_ch_x)

        out = tf.concat(out, axis=-1)

        if self.normalize is not None:
            out = tf.math.divide(tf.math.subtract(out, self.norm_mean), self.norm_std)

        return out
 

    def _get_freq_indices(self, freq_component_set, freq_component_selection_of_each_channel, num_selected_freq_component_of_each_channel):
        component_number = 'component_number'
        freq_u_idx = 'freq_u_idx'
        freq_v_idx = 'freq_v_idx'
        # all 64 frequency components
        freq_64_components = [
            {component_number: 0, freq_u_idx: 0, freq_v_idx: 0},
            {component_number: 1, freq_u_idx: 1, freq_v_idx: 0},
            {component_number: 2, freq_u_idx: 0, freq_v_idx: 1},
            {component_number: 3, freq_u_idx: 0, freq_v_idx: 2},
            {component_number: 4, freq_u_idx: 1, freq_v_idx: 1},
            {component_number: 5, freq_u_idx: 2, freq_v_idx: 0},
            {component_number: 6, freq_u_idx: 3, freq_v_idx: 0},
            {component_number: 7, freq_u_idx: 2, freq_v_idx: 1},
            {component_number: 8, freq_u_idx: 1, freq_v_idx: 2},
            {component_number: 9, freq_u_idx: 0, freq_v_idx: 3},
            {component_number: 10, freq_u_idx: 0, freq_v_idx: 4},
            {component_number: 11, freq_u_idx: 1, freq_v_idx: 3},
            {component_number: 12, freq_u_idx: 2, freq_v_idx: 2},
            {component_number: 13, freq_u_idx: 3, freq_v_idx: 1},
            {component_number: 14, freq_u_idx: 4, freq_v_idx: 0},
            {component_number: 15, freq_u_idx: 5, freq_v_idx: 0},
            {component_number: 16, freq_u_idx: 4, freq_v_idx: 1},
            {component_number: 17, freq_u_idx: 3, freq_v_idx: 2},
            {component_number: 18, freq_u_idx: 2, freq_v_idx: 3},
            {component_number: 19, freq_u_idx: 1, freq_v_idx: 4},
            {component_number: 20, freq_u_idx: 0, freq_v_idx: 5},
            {component_number: 21, freq_u_idx: 0, freq_v_idx: 6},
            {component_number: 22, freq_u_idx: 1, freq_v_idx: 5},
            {component_number: 23, freq_u_idx: 2, freq_v_idx: 4},
            {component_number: 24, freq_u_idx: 3, freq_v_idx: 3},
            {component_number: 25, freq_u_idx: 4, freq_v_idx: 2},
            {component_number: 26, freq_u_idx: 5, freq_v_idx: 1},
            {component_number: 27, freq_u_idx: 6, freq_v_idx: 0},
            {component_number: 28, freq_u_idx: 7, freq_v_idx: 0},
            {component_number: 29, freq_u_idx: 6, freq_v_idx: 1},
            {component_number: 30, freq_u_idx: 5, freq_v_idx: 2},
            {component_number: 31, freq_u_idx: 4, freq_v_idx: 3},
            {component_number: 32, freq_u_idx: 3, freq_v_idx: 4},
            {component_number: 33, freq_u_idx: 2, freq_v_idx: 5},
            {component_number: 34, freq_u_idx: 1, freq_v_idx: 6},
            {component_number: 35, freq_u_idx: 0, freq_v_idx: 7},
            {component_number: 36, freq_u_idx: 1, freq_v_idx: 7},
            {component_number: 37, freq_u_idx: 2, freq_v_idx: 6},
            {component_number: 38, freq_u_idx: 3, freq_v_idx: 5},
            {component_number: 39, freq_u_idx: 4, freq_v_idx: 4},
            {component_number: 40, freq_u_idx: 5, freq_v_idx: 3},
            {component_number: 41, freq_u_idx: 6, freq_v_idx: 2},
            {component_number: 42, freq_u_idx: 7, freq_v_idx: 1},
            {component_number: 43, freq_u_idx: 7, freq_v_idx: 2},
            {component_number: 44, freq_u_idx: 6, freq_v_idx: 3},
            {component_number: 45, freq_u_idx: 5, freq_v_idx: 4},
            {component_number: 46, freq_u_idx: 4, freq_v_idx: 5},
            {component_number: 47, freq_u_idx: 3, freq_v_idx: 6},
            {component_number: 48, freq_u_idx: 2, freq_v_idx: 7},
            {component_number: 49, freq_u_idx: 3, freq_v_idx: 7},
            {component_number: 50, freq_u_idx: 4, freq_v_idx: 6},
            {component_number: 51, freq_u_idx: 5, freq_v_idx: 5},
            {component_number: 52, freq_u_idx: 6, freq_v_idx: 4},
            {component_number: 53, freq_u_idx: 7, freq_v_idx: 3},
            {component_number: 54, freq_u_idx: 7, freq_v_idx: 4},
            {component_number: 55, freq_u_idx: 6, freq_v_idx: 5},
            {component_number: 56, freq_u_idx: 5, freq_v_idx: 6},
            {component_number: 57, freq_u_idx: 4, freq_v_idx: 7},
            {component_number: 58, freq_u_idx: 5, freq_v_idx: 7},
            {component_number: 59, freq_u_idx: 6, freq_v_idx: 6},
            {component_number: 60, freq_u_idx: 7, freq_v_idx: 5},
            {component_number: 61, freq_u_idx: 7, freq_v_idx: 6},
            {component_number: 62, freq_u_idx: 6, freq_v_idx: 7},
            {component_number: 63, freq_u_idx: 7, freq_v_idx: 7}
        ]

        # defines frequency component set used
        assert freq_component_set in [64]
        if freq_component_set == 64:
            all_freq_component = freq_64_components
        else:
            raise NotImplementedError

        # defines selected frequency components for each channel
        self.selected_freq_for_channels = dict()
        if freq_component_selection_of_each_channel['format'] == 'freq_code': # need to convert 'freq_code' to list of 'freq_number'
            all_freq_component_num = [*range(freq_component_set)]
            for ch_num, freq_code_and_config_of_ch in freq_component_selection_of_each_channel['channel'].items():
                freq_code_of_ch = freq_code_and_config_of_ch['code']
                freq_config_of_ch = freq_code_and_config_of_ch['config'] 
                if freq_code_of_ch == 'all':
                    self.selected_freq_for_channels[ch_num] = all_freq_component_num
                elif freq_code_of_ch == 'low':
                    num_freq = int(freq_config_of_ch)
                    assert num_freq <= freq_component_set
                    self.selected_freq_for_channels[ch_num] = all_freq_component_num[:num_freq]
                elif freq_code_of_ch == 'high':
                    num_freq = int(freq_config_of_ch)
                    assert num_freq <= freq_component_set
                    self.selected_freq_for_channels[ch_num] = all_freq_component_num[-num_freq:]
                elif freq_code_of_ch == 'step':
                    step_size = int(freq_config_of_ch)
                    self.selected_freq_for_channels[ch_num] = all_freq_component_num[::step_size]
                elif freq_code_of_ch == 'band':
                    num_band = int(freq_config_of_ch[0])
                    num_freq_of_each_band = int(freq_config_of_ch[1])
                    assert num_freq_of_each_band <= (freq_component_set // num_band)
                    self.selected_freq_for_channels[ch_num] = list()
                    for band_num in range(num_band):
                        start_freq_of_this_band = band_num * (freq_component_set // num_band)
                        selected_freq_of_this_band = [*range(start_freq_of_this_band, start_freq_of_this_band + num_freq_of_each_band)]
                        self.selected_freq_for_channels[ch_num].extend(selected_freq_of_this_band)
                else:
                    raise NotImplementedError
        elif freq_component_selection_of_each_channel['format'] == 'freq_number':
            self.selected_freq_for_channels = freq_component_selection_of_each_channel['channel']
        else:
            raise NotImplementedError
        
        # find freq_idxs_u and freq_idxs_v of selected frequency components for each channel
        freq_idxs_u = dict()
        freq_idxs_v = dict()
        for ch_num, list_selected_freq_of_ch in self.selected_freq_for_channels.items():
            freq_idxs_u[ch_num] = [f[freq_u_idx] for f in all_freq_component if f[component_number] in list_selected_freq_of_ch]
            freq_idxs_v[ch_num] = [f[freq_v_idx] for f in all_freq_component if f[component_number] in list_selected_freq_of_ch]

        # verify whether number of selected frequency for each channel between 'freq_idxs_u/v' and 'num_selected_freq_component_of_each_channel' is equal
        for ch_num, num_selected_freq_of_ch in enumerate(num_selected_freq_component_of_each_channel):
            assert len(freq_idxs_u[ch_num]) == num_selected_freq_of_ch
            assert len(freq_idxs_v[ch_num]) == num_selected_freq_of_ch

        return freq_idxs_u, freq_idxs_v
    

    def _get_dct_filter(self, tile_size_x, tile_size_y, freq_idxs_u, freq_idxs_v):
        assert tile_size_x == tile_size_y
        assert len(freq_idxs_u) == len(freq_idxs_v)
        
        num_selected_freq = len(freq_idxs_u)

        dct_filter = np.zeros(shape=[tile_size_x, tile_size_y, num_selected_freq], dtype=np.float32)

        for f_i, (f_u, f_v) in enumerate(zip(freq_idxs_u, freq_idxs_v)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[t_x, t_y, f_i] = self._build_dct_filter(t_x, f_u, tile_size_x) * self._build_dct_filter(t_y, f_v, tile_size_y)

        # prepares 'dct_filter' for 'tf.nn.conv2d' operation in each corresponding input channel. 
        # as 'tf.nn.conv2d' need 4-D filters tensor of shape [filter_height, filter_width, in_channels, out_channels], 
        # we need to expand_dims at axis 2 (with length = 1 -> one in_channel) for 'dct_filter' 
        # to change shape from [tile_size_x, tile_size_y, num_selected_freq] to [tile_size_x, tile_size_y, 1, num_selected_freq]
        dct_filter = tf.expand_dims(dct_filter, axis=2)

        return dct_filter


    def _build_dct_filter(self, pos, freq, size):
        result = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)