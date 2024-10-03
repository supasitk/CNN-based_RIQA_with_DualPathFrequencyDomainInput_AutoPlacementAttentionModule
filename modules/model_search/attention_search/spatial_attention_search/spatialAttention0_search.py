import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, GlobalMaxPool2D)

from modules.labels import *

##################################################################################################
# Support SEARCH options: 'kernel_size' (list of integer)                                        #
##################################################################################################
# Default STATIC options: filters=1, stride=1, activation='softplus', epsilon=0.1                #
##################################################################################################

class SpatialAttention0_Search(tf.keras.layers.Layer):
    """Spatial Attenton block from paper 'Learning for retinal image quality assessment with label regularization [2022]'."""
    def __init__(self, block_id, kernel_size_search_infos, filters=1, stride=1, activation='softplus', epsilon=0.1, name=None, **kwargs):
        super(SpatialAttention0_Search, self).__init__(name=name, **kwargs)
        self.name = name
        self.block_id = block_id

        # ----------- Conv Operations for searching ------------ #
        # get list of operations of Conv as list of kernel_size search options specified 
        # and start weight index of kernel_size search options
        list_of_kernel_sizes = kernel_size_search_infos[label_of_value_of_search_option]
        self.start_weight_idx_of_kernel_sizes = kernel_size_search_infos[label_of_start_weight_idx_of_search_option]

        self.conv_ops_with_kernel_sizes = list()

        for this_kernel_size in list_of_kernel_sizes:
            conv_ops_with_this_kernel_size = Conv2D(filters=filters,
                                                    kernel_size=this_kernel_size,
                                                    strides=stride,
                                                    padding="SAME",
                                                    activation=activation,
                                                    name=name + "_conv_k" + str(this_kernel_size))
            
            self.conv_ops_with_kernel_sizes.append(conv_ops_with_this_kernel_size)
        
        self.epsilon = epsilon
        self.globalMaxPool = GlobalMaxPool2D(keepdims=True)


    def call(self, inputs, alphas_intra_attn_weights, training=False):
        # ----------- Conv Operations for searching ------------ #
        out_conv_ops_with_kernel_sizes = list()
        for conv_op_id, conv_op in enumerate(self.conv_ops_with_kernel_sizes):
            out_conv_op = conv_op(inputs)
            out_conv_op = out_conv_op * alphas_intra_attn_weights[self.block_id, self.start_weight_idx_of_kernel_sizes + conv_op_id]

            out_conv_ops_with_kernel_sizes.append(out_conv_op)
        
        mix_out_conv_ops_with_kernel_sizes = tf.add_n(out_conv_ops_with_kernel_sizes)
        
        out_add_epsilon = tf.math.add(mix_out_conv_ops_with_kernel_sizes, self.epsilon)
      
        max = self.globalMaxPool(out_add_epsilon)

        attn = tf.math.divide(out_add_epsilon, max)

        return tf.math.multiply(inputs, attn)
