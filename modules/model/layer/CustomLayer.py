import tensorflow as tf


class BranchInput(tf.keras.layers.Layer):
    """Gather input for each branch of model's structure according to specific indices of input channel"""
    def __init__(self, Branch_idx_input_ch, axis=-1, name=None, **kwargs):
        super(BranchInput, self).__init__(name=name, **kwargs)
        self.Branch_idx_input_ch = Branch_idx_input_ch
        self.axis = axis
    
    def call(self, inputs, training=False):
        return tf.gather(inputs, self.Branch_idx_input_ch, axis=self.axis)
