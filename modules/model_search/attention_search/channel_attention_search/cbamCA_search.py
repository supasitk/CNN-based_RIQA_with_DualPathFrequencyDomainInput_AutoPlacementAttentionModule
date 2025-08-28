import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Reshape, ZeroPadding1D, Activation)

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
#                    Squeeze op (MAX_pool) ||   | Squeeze op (AVG_pool)                          #
#                                          ||   |                                                #
#                                          ||   |                                                #
#                                          ||   | Exciation_1st (Dense_1st) op                   #
#                                          ||   | with search option as list of 'reduction_ratio'#
#                                          vv   v                                                #
#                   -----------------------------------------------------                        # 
#                   ||    |                  ||    |                ||  |                        #                    
#                   vv    v                  vv    v                vv  v                        #
#              -------------------    --------------------   ---------------------               #
#              | Dense_r_ratio#0 |    | Dense_r_ratio#.. |   | Dense_r_ratio#N-1 |               #
#              -------------------    --------------------   ---------------------               # 
#                   ||    |                  ||    |                   ||    |                   #
#           ------------------------    ------------------------   ------------------------      #
#           |     ZeroPadding as   |    |     ZeroPadding as   |   |     ZeroPadding as   |      #
#           | MIN(r_ratio_options) |    | MIN(r_ratio_options) |   | MIN(r_ratio_options) |      #
#           |     output size      |    |     output size      |   |     output size      |      #  
#           ------------------------    ------------------------   ------------------------      # 
#                       ||  |                       ||  |                       ||   |           #
#     * alpha_r_ratio#0 ||  v    * alpha_r_ratio#.. ||  v   * alpha_r_ratio#N-1 ||   v           #
#                       ||  ------------------> + <-||--------------------------||----           #
#                       vv                      |   vv                          vv               # 
#                       ===============> + <====|=================================               #
#                                        ||     |                                                #
#                                        vv     v                                                #
#                                    -----------------------                                     #
#                                    | Excitation_1st act  |                                     #
#                                    |        (ReLU)       |                                     #
#                                    -----------------------                                     #
#                                             ||  |                                              #
#                                             vv  v                                              #
#                               ---------------------------------                                #
#                               | Excitation_2nd (Dense_2nd) op |                                #
#                               ---------------------------------                                #
#                                           ||      |                                            #
#                                           vv      v                                            #
#                                           --> + <--                                            #
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

        # -------- Squeeze Operation(s) ---------- #
        self.squeeze_ops = list()
        for spatial_pooling_method in spatial_pooling_methods:
            if spatial_pooling_method == 'avg':
                spatial_pooling_op = GlobalAveragePooling2D(name=f'{self.name}_spatialpool_avg')
            elif spatial_pooling_method == 'max':
                spatial_pooling_op = GlobalMaxPool2D(name=f'{name}_spatialpool_max')
            else:
                raise ValueError('Unknown method \'{}\' for SpatialPooling layer'.format(spatial_pooling_method))
            self.squeeze_ops.append(spatial_pooling_op)

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
                                                                use_bias=use_bias,
                                                                seed=self.seed,
                                                                name=self.name)
            self.dense1_ops_with_r_ratios.append(dense1_ops_with_this_r_ratio)

        # -------- Exciation_1st Activation ---------- #
        self.dense1_act = Activation(activation=activation_dense1,
                                     name=name + "_dense1_act")

        # -------- Exciation_2nd (Dense_2nd) Opeartion ---------- #
        self.dense2_op = Dense(units=self.ch_out,
                               activation=None,
                               use_bias=use_bias,
                               kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                               name=name + "_dense2_op")

        # -------- Exciation_2nd Activation ---------- #
        self.dense2_act = Activation(activation=activation_dense2,
                                     name=name + "_dense2_act")


    def _get_dense1_ops(self, min_r_ratios, this_r_ratio, use_bias=True, seed=None, name=None):
        # calculate how many zeros to add at the beginning and end of the padding dimension (axis 1)
        num_left_pad = 0
        num_right_pad = (self.ch_out // min_r_ratios) - (self.ch_out // this_r_ratio)

        return Sequential([Dense(units=self.ch_out // this_r_ratio,
                                 use_bias=use_bias,
                                 kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                                 name=name + "_dense1_op_r" + str(this_r_ratio)),
                           Reshape([-1, 1]),
                           ZeroPadding1D(padding=(num_left_pad, num_right_pad)),
                           Reshape([-1])                        
                          ]
                         )


    def call(self, inputs, alphas_intra_attn_weights, training=False):
        # -------- Squeeze Operation ---------- #
        # output: list of squeeze operation(s) -> [out_squeeze_op#1, out_squeeze_op#2, ....]
        out_squeeze_ops = list()
        for squeeze_op in self.squeeze_ops:
            out_squeeze_ops.append(squeeze_op(inputs))
        
        # -------- Exciation_1st (Dense_1st) Opeartions for searching ---------- #
        # output: nested-list of dense1_op_with_r_ratio operation(s) -> [
        #                                                                [out_dense1_op_with_r#1_for_out_squeeze_op#1, out_dense1_op_with_r#1_for_out_squeeze_op#2, ...], 
        #                                                                [out_dense1_op_with_r#2_for_out_squeeze_op#1, out_dense1_op_with_r#2_for_out_squeeze_op#2, ...], 
        #                                                                [...]    
        #                                                               ]
        out_dense1_ops_with_r_ratios_for_out_squeeze_ops = list()
        for dense1_op_id, dense1_op in enumerate(self.dense1_ops_with_r_ratios):
            out_dense1_op_for_out_squeeze_ops = list()
            for out_squeeze_op in out_squeeze_ops:
                out_dense1_op_for_out_squeeze_op = dense1_op(out_squeeze_op)
                out_dense1_op_for_out_squeeze_op = out_dense1_op_for_out_squeeze_op * alphas_intra_attn_weights[self.block_id, self.start_weight_idx_of_r_ratios + dense1_op_id]
                out_dense1_op_for_out_squeeze_ops.append(out_dense1_op_for_out_squeeze_op)
            out_dense1_ops_with_r_ratios_for_out_squeeze_ops.append(out_dense1_op_for_out_squeeze_ops)

        # mix different output of dense1_op_with_r_ratio operation(s) for each same out_squeeze_op#n by
        #   nested-list of dense1_op_with_r_ratio operation(s) delisted and then zipping to get 
        #   tuple of (out_dense1_op_with_r#1_for_out_squeeze_op#1, out_dense1_op_with_r#2_for_out_squeeze_op#1, ...), 
        #            (out_dense1_op_with_r#1_for_out_squeeze_op#2, out_dense1_op_with_r#2_for_out_squeeze_op#2, ...),
        #            ...
        #   each tuple converted to list and add its elements together
        # output: list of mix_out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_ops -> [mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#1, mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#2, ...]
        mix_out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_ops = [tf.math.add_n(list(out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_op)) for out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_op in zip(*out_dense1_ops_with_r_ratios_for_out_squeeze_ops)]

        # -------- Exciation_1st Activation ---------- #
        # output: list of dense1_activation operation -> [out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#1, out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#2, ....]
        out_dense1_acts = list()
        for mix_out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_op in mix_out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_ops:
            out_dense1_acts.append(self.dense1_act(mix_out_dense1_ops_with_r_ratios_for_each_same_out_squeeze_op))

        # -------- Exciation_2nd (Dense_2nd) Opeartion ---------- #
        # output: list of dense2 operation -> [out_dense2_op_of_out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#1, out_dense2_op_of_out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#2, ....]
        out_dense2_ops = list()
        for out_dense1_act in out_dense1_acts:
            out_dense2_ops.append(self.dense2_op(out_dense1_act))

        # add every out_dense2_op_of_out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#N together
        # output: sum_of_out_dense2_op_of_out_dense1_act_of_mix_out_dense1_op_with_r_ratio(s)_for_out_squeeze_op#N
        out_dense2_ops_sum = tf.math.add_n(out_dense2_ops)

        # -------- Exciation_2nd Activation ---------- #
        attn = self.dense2_act(out_dense2_ops_sum)
        
        attn = tf.reshape(attn, [-1, 1, 1, self.ch_out])
        return tf.math.multiply(inputs, attn)