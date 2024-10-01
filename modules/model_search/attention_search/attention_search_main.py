import tensorflow as tf

from modules.labels import *
######### TEST ##########
from modules.model.layer.CustomLayer import (Skip)
from modules.model.attention.channel_attention.se import (ChannelAttention0)
from modules.model.attention.spatial_attention.spatialAttention0 import (SpatialAttention0)
######### TEST ##########

class AttentionSearchModule(tf.keras.layers.Layer):
    """Module contains operations of searching inter- and intra-attention types for the model"""
    def __init__(self, block_id, numChannel_InputFeature, attention, name=None, **kwargs):
        super(AttentionSearchModule, self).__init__(name=name, **kwargs)
        self.block_id = block_id
        self.numChannel_InputFeature = numChannel_InputFeature
        self.attention = attention
        self.name = name

        # get list of attention types for INTER-attention_types searching to find which attention type module is the optimized option 
        # (selecting the optimized attention type among all attention types searched) for attaching to this block
        list_attn_types_for_searching = attention[label_of_inter_search_infos][label_of_list_of_attn_type]
        
        # get operations of INTRA-attention type searching for each of attention types 
        # to find which configurations of each attention type is the optimized option for attaching to this block
        self._attn_types_ops = []
        for attn_type in list_attn_types_for_searching:
            ######### TEST ##########
            if attn_type == 'skip':
                op = Skip()
            elif attn_type == 'ChannelAttention0':
                op = ChannelAttention0(channel_output=self.numChannel_InputFeature,
                                       reduction_ratio=16, 
                                       activation_dense1='relu', 
                                       activation_dense2='sigmoid',
                                       name=name)
            elif attn_type == 'SpatialAttention0':
                op = SpatialAttention0(filters=1,
                                       kernel_size=1,
                                       stride=1,
                                       activation='softplus',
                                       epsilon=0.1,
                                       name=name)

            self._attn_types_ops.append(op)
            ######### TEST ##########

    ######### TEST ##########
    def call(self, x, alphas_inter_attn_weights, alphas_intra_attn_weights, training=False):
        out = x
        op_outs = []
        for op_id, op in enumerate(self._attn_types_ops):
            op_out = op(out)
            op_out = op_out * alphas_inter_attn_weights[self.block_id, op_id]
            op_outs.append(op_out)
        mix_op_outs = tf.add_n(op_outs)

        return mix_op_outs
    ######### TEST ##########