import tensorflow as tf
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense)
 

class ChannelAttention0_Search(tf.keras.layers.Layer):
    """Channel Attenton block from paper 'Squeeze-and-Excitation (SE) Networks [2017]'."""
    def __init__(self, block_id, channel_output, reduction_ratio_search_infos, activation_dense1='relu', activation_dense2='sigmoid', name=None, **kwargs):
        super(ChannelAttention0_Search, self).__init__(name=name, **kwargs)
        self.ch_out = channel_output
        self.squeeze = GlobalAveragePooling2D()
        ### TEST ###
        print('block {}: reduction_ratio_search_infos: {}'.format(block_id, reduction_ratio_search_infos))
        reduction_ratio = 16
        ### TEST ###
        self.dense1 = Dense(units=channel_output // reduction_ratio,
                            activation=activation_dense1,
                            name=name + "_dense1")
        self.dense2 = Dense(units=channel_output,
                            activation=activation_dense2,
                            name=name + "_dense2")
    
    def call(self, inputs, alphas_intra_attn_weights, training=False):
        attn = self.squeeze(inputs)
        attn = self.dense1(attn)
        attn = self.dense2(attn)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(inputs, attn)