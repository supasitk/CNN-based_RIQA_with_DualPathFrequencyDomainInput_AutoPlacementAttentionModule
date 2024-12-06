import math
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (Softmax)


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
    

class DCT2d(tf.keras.layers.Layer):
    """Generate layer of 2D-DCT filters and transforming input into 2D-DCT coefficients"""
    def __init__(self, num_freq=64, freq_selection=None, tile_size_x=8, tile_size_y=8, name='DCT2d', **kwargs):
        super(DCT2d, self).__init__(name=name, **kwargs)
        self.num_freq = num_freq
        self.freq_selection = freq_selection 
        self.tile_size_x = tile_size_x 
        self.tile_size_y = tile_size_y
        self.name = name
    

    def build(self, input_shape):
        self.in_channel = input_shape[-1]
        
        freq_idxs_u, freq_idxs_v = self._get_freq_indices(self.num_freq, self.freq_selection)
        # fixed DCT filters init
        self.weight = tf.Variable(initial_value=self._get_dct_filter(self.tile_size_x, self.tile_size_y, freq_idxs_u, freq_idxs_v, self.in_channel),
                                  trainable=False)
        
    
    def call(self, x):
        out = tf.nn.depthwise_conv2d(x, self.weight, strides=[1, self.tile_size_x, self.tile_size_y, 1], padding='VALID')

        return out


    def _get_freq_indices(self, num_freq, freq_selection):
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

        assert num_freq in [64]
        assert freq_selection in [None]

        if num_freq == 64:
            all_freq_component_num = freq_64_components
        else:
            raise NotImplementedError

        if freq_selection is None: # No frequency selection -> all frequency used
            selected_freq_component_num = [*range(num_freq)]
            freq_idxs_u = [f[freq_u_idx] for f in all_freq_component_num if f[component_number] in selected_freq_component_num]
            freq_idxs_v = [f[freq_v_idx] for f in all_freq_component_num if f[component_number] in selected_freq_component_num]
        else:
            raise NotImplementedError

        return freq_idxs_u, freq_idxs_v
    

    def _get_dct_filter(self, tile_size_x, tile_size_y, freq_idxs_u, freq_idxs_v, in_channel):
        assert tile_size_x == tile_size_y
        assert len(freq_idxs_u) == len(freq_idxs_v)
        
        num_selected_freq = len(freq_idxs_u)

        dct_filter = np.zeros(shape=[tile_size_x, tile_size_y, num_selected_freq], dtype=np.float32)

        for f_i, (f_u, f_v) in enumerate(zip(freq_idxs_u, freq_idxs_v)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[t_x, t_y, f_i] = self._build_dct_filter(t_x, f_u, tile_size_x) * self._build_dct_filter(t_y, f_v, tile_size_y)
        
        # duplicate dct filter for each input channel to seperately transform each input channel into its 2D-DCT coefficients
        list_freq_idx = list()
        for f_i in range(num_selected_freq):
            list_freq_idx.extend([f_i]*in_channel)
        dct_filters = tf.gather(dct_filter, list_freq_idx, axis=-1)
        dct_filters = tf.reshape(dct_filters, [tile_size_x, tile_size_y, num_selected_freq, in_channel])
        dct_filters = tf.transpose(dct_filters, perm=[0, 1, 3, 2])

        return dct_filters        


    def _build_dct_filter(self, pos, freq, size):
        result = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)