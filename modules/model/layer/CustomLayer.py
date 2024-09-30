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
    
    def call(self, value):
        splitsotfmax_values = []
        for block_id in range(self.num_blocks):
            splitsotfmax_values.append(tf.concat([self.soft_max_func(t) for t in tf.split(value[block_id], self.size_splits)], axis=0))
        
        return tf.reshape(splitsotfmax_values, [self.num_blocks, self.num_ops])