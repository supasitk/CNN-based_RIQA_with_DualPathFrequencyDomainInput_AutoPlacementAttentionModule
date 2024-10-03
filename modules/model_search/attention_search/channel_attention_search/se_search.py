import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Reshape, ZeroPadding1D, ReLU)

from modules.labels import *

##################################################################################################
# Support SEARCH options: 'reduction_ratio' (list of integer)                                    #
##################################################################################################
# Default STATIC options: activation_dense1='relu', activation_dense2='sigmoid'                  #
##################################################################################################
# Detail of Searching options of SE block                                                        # 
#                                                                                                #
#                                    -----------------------                                     #
#                                    | Input of Attn block |                                     # 
#                                    -----------------------                                     #     
#                                               |                                                # 
#                                    -----------------------                                     #
#                                    |      Squeeze op     |                                     #
#                                    |  (global_AVG_pool)  |                                     #
#                                    -----------------------                                     #
#                                               |                                                #
#                                               | Exciation_1st (Dense_1st) op                   #
#                                               | with search option as list of 'reduction_ratio'#
#                                               v                                                #
#                       -----------------------------------------------                          # 
#                       |                       |                     |                          #                    
#                       v                       v                     v                          #
#              -------------------    --------------------   -------------------                 #
#              | Dense_r_ratio#1 |    | Dense_r_ratio#.. |   | Dense_r_ratio#N |                 #
#              -------------------    --------------------   -------------------                 # 
#                       |                       |                     |                          #
#           ------------------------    ------------------------   ------------------------      #
#           |     ZeroPadding as   |    |     ZeroPadding as   |   |     ZeroPadding as   |      #
#           | MIN(r_ratio_options) |    | MIN(r_ratio_options) |   | MIN(r_ratio_options) |      #
#           |     output size      |    |     output size      |   |     output size      |      #  
#           ------------------------    ------------------------   ------------------------      # 
#                       |                       |                              |                 #
#     * alpha_r_ratio#1 |    * alpha_r_ratio#.. v            * alpha_r_ratio#N |                 #
#                       ----------------------> + <-----------------------------                 #
#                                               |                                                #
#                                               v                                                #
#                                    -----------------------                                     #
#                                    | Excitation_1st act  |                                     #
#                                    |        (ReLU)       |                                     #
#                                    -----------------------                                     #
#                                               |                                                #
#                                               v                                                #
#                               ---------------------------------                                #
#                               | Excitation_2nd (Dense_2nd) op |                                #
#                               ---------------------------------                                #
#                                               |                                                #
#                                               v                                                #
#                                    -----------------------                                     #
#                                    | Excitation_2nd act  |                                     #
#                                    |      (Sigmoid)      |                                     #
#                                    -----------------------                                     #
#                                               |                                                #
#                                               v                                                #
#                                    -----------------------                                     #
#                                    |    Attention map    |                                     #
#                                    -----------------------                                     #
#                                               |                                                #
#                                               v Multiply with Input of Attn block              #
#                                                                                                # 
##################################################################################################


class ChannelAttention0_Search(tf.keras.layers.Layer):
    """Channel Attenton block from paper 'Squeeze-and-Excitation (SE) Networks [2017]'."""
    def __init__(self, block_id, channel_output, reduction_ratio_search_infos, activation_dense1='relu', activation_dense2='sigmoid', name=None, **kwargs):
        super(ChannelAttention0_Search, self).__init__(name=name, **kwargs)
        self.ch_out = channel_output

        # -------- Squeeze Operation ---------- #
        self.squeeze = GlobalAveragePooling2D()

        # -------- Exciation_1st (Dense_1st) Opeartions for searching ---------- #
        # get list of operations of Excitation_1st (Dense_1st) as list of reduction_ratio search options specified 
        # and start weight index of reduction_ratio search options 
        list_of_r_ratios = reduction_ratio_search_infos[label_of_value_of_search_option]
        self.start_weight_idx_of_r_ratios = reduction_ratio_search_infos[label_of_start_weight_idx_of_search_option] 
        
        min_value_of_r_ratios = min(list_of_r_ratios)

        self.dense1_ops_with_r_ratios = list()

        for this_r_ratio in list_of_r_ratios:
            dense1_ops_with_this_r_ratio = self._get_dense1_ops(min_r_ratios=min_value_of_r_ratios, 
                                                                this_r_ratio=this_r_ratio)

            self.dense1_ops_with_r_ratios.append(dense1_ops_with_this_r_ratio)

        # -------- Exciation_1st Activation for searching ---------- #
        if activation_dense1 == 'relu':
            dense1_act_func = ReLU

        self.dense1_act = dense1_act_func(name=name + "_dense1_act")

        # -------- Exciation_2nd (Dense_2nd) Opeartion and Activation ---------- #
        self.dense2 = Dense(units=channel_output,
                            activation=activation_dense2,
                            name=name + "_dense2_op_act")


    def call(self, inputs, alphas_intra_attn_weights, training=False):
        attn = self.squeeze(inputs)
        attn = self.dense1(attn)
        attn = self.dense2(attn)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(inputs, attn)


    def _get_dense1_ops(self, min_r_ratios, this_r_ratio):
        # calculate how many zeros to add at the beginning and end of the padding dimension (axis 1)
        num_left_pad = 0
        num_right_pad = (self.ch_out // min_r_ratios) - (self.ch_out // this_r_ratio)

        return Sequential([Dense(units=self.ch_out // this_r_ratio,
                                 name=name + "_dense1_op"),
                           Reshape([-1, 1]),
                           ZeroPadding1D(padding=(num_left_pad, num_right_pad)),
                           Reshape([-1])                        
                          ]
                         )