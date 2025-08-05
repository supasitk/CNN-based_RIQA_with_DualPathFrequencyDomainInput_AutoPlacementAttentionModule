import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Reshape, ZeroPadding1D, ReLU)

from modules.labels import *
from modules.initializers import *

##################################################################################################
# Support SEARCH options: 'reduction_ratio' (list of integer)                                    #
##################################################################################################
# Default STATIC options: spatial_pooling_methods=['avg', 'max'], use_bias=True,                 #
#                         activation_dense1='relu', activation_dense2='sigmoid'                  #
##################################################################################################
# Detail of Searching options of CBAM_ChannelAttention block                                     # 
#                                                                                                #
#                                    -----------------------                                     #
#                                    | Input of Attn block |                                     # 
#                                    -----------------------                                     #     
#                                               |                                                # 
#                                    -----------------------                                     #
#                                    |      Squeeze op     |                                     #
#                                    |  (global_AVG_pool)  |                                     #
#                                    |         and         |                                     #
#                                    |  (global_MAX_pool)  |                                     #
#                                    -----------------------                                     #
#                                               |                                                #
#                                               | Exciation_1st (Dense_1st) op                   #
#                                               | with search option as list of 'reduction_ratio'#
#                                               v                                                #
#                       -----------------------------------------------                          # 
#                       |                       |                     |                          #                    
#                       v                       v                     v                          #
#              -------------------    --------------------   ---------------------               #
#              | Dense_r_ratio#0 |    | Dense_r_ratio#.. |   | Dense_r_ratio#N-1 |               #
#              -------------------    --------------------   ---------------------               # 
#                       |                       |                     |                          #
#           ------------------------    ------------------------   ------------------------      #
#           |     ZeroPadding as   |    |     ZeroPadding as   |   |     ZeroPadding as   |      #
#           | MIN(r_ratio_options) |    | MIN(r_ratio_options) |   | MIN(r_ratio_options) |      #
#           |     output size      |    |     output size      |   |     output size      |      #  
#           ------------------------    ------------------------   ------------------------      # 
#                       |                       |                              |                 #
#     * alpha_r_ratio#0 |    * alpha_r_ratio#.. v          * alpha_r_ratio#N-1 |                 #
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
#                                               v Multiply Input of Attn block with attn map     #
#                                                                                                # 
##################################################################################################


class ChannelAttention1_Search(tf.keras.layers.Layer):
    """Channel Attenton block from paper 'CBAM: Convolutional Block Attention Module [2018]'."""
    def __init__(self, block_id, channel_output, 
                 reduction_ratio_search_infos, spatial_pooling_methods=['avg', 'max'], use_bias=True, activation_dense1='relu', activation_dense2='sigmoid', 
                 seed=None, name=None, **kwargs):
        super(ChannelAttention1_Search, self).__init__(name=name, **kwargs)
        self.seed = seed
        self.name = name
        self.block_id = block_id
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
                                                                this_r_ratio=this_r_ratio,
                                                                seed=self.seed,
                                                                name=self.name)

            self.dense1_ops_with_r_ratios.append(dense1_ops_with_this_r_ratio)

        # -------- Exciation_1st Activation ---------- #
        if activation_dense1 == 'relu':
            dense1_act_func = ReLU

        self.dense1_act = dense1_act_func(name=name + "_dense1_act")

        # -------- Exciation_2nd (Dense_2nd) Opeartion and Activation ---------- #
        self.dense2 = Dense(units=channel_output,
                            activation=activation_dense2,
                            kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                            name=name + "_dense2_op_act")


    def call(self, inputs, alphas_intra_attn_weights, training=False):
        # -------- Squeeze Operation ---------- #
        out_squeeze_op = self.squeeze(inputs)

        # -------- Exciation_1st (Dense_1st) Opeartions for searching ---------- #
        out_dense1_ops_with_r_ratios = list()
        for dense1_op_id, dense1_op in enumerate(self.dense1_ops_with_r_ratios):
            out_dense1_op = dense1_op(out_squeeze_op)
            out_dense1_op = out_dense1_op * alphas_intra_attn_weights[self.block_id, self.start_weight_idx_of_r_ratios + dense1_op_id]

            out_dense1_ops_with_r_ratios.append(out_dense1_op)
        
        mix_out_dense1_ops_with_r_ratios = tf.add_n(out_dense1_ops_with_r_ratios)

        # -------- Exciation_1st Activation ---------- #
        out_dense1_act = self.dense1_act(mix_out_dense1_ops_with_r_ratios)

        # -------- Exciation_2nd (Dense_2nd) Opeartion and Activation ---------- #
        attn = self.dense2(out_dense1_act)
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(inputs, attn)


    def _get_dense1_ops(self, min_r_ratios, this_r_ratio, seed=None, name=None):
        # calculate how many zeros to add at the beginning and end of the padding dimension (axis 1)
        num_left_pad = 0
        num_right_pad = (self.ch_out // min_r_ratios) - (self.ch_out // this_r_ratio)

        return Sequential([Dense(units=self.ch_out // this_r_ratio,
                                 kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                                 name=name + "_dense1_op_r" + str(this_r_ratio)),
                           Reshape([-1, 1]),
                           ZeroPadding1D(padding=(num_left_pad, num_right_pad)),
                           Reshape([-1])                        
                          ]
                         )