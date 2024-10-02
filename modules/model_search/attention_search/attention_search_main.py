import tensorflow as tf

from modules.labels import *
from modules.model.layer.CustomLayer import (Skip)
from modules.model_search.attention_search.channel_attention_search.se_search import (ChannelAttention0_Search)
from modules.model_search.attention_search.spatial_attention_search.spatialAttention0_search import (SpatialAttention0_Search)


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
        # however, before INTER-attention_types searching, operations of INTRA-attention type searching for each of attention types 
        # need to be first processed to find which configurations of each attention type is the optimized option for each one and then 
        # one of the fine-tuned attention types (by INTRA-attention type searching) selected by INTER-attention_types searching for attaching to this block
        list_attn_types_for_searching = attention[label_of_inter_search_infos][label_of_list_of_attn_type]
                
        self.intra_ops_of_attn_types = list()
        
        for attn_type in list_attn_types_for_searching:
            if attn_type == 'skip': # no attention
                intra_ops_of_this_attn_type = Skip()
            
            elif attn_type == 'ChannelAttention0': # SE
                # support search options: 'reduction_ratio' (list of integer)
                # default static options: activation_dense1='relu', activation_dense2='sigmoid'

                # 'reduction_ratio' search infos
                r_ratio = 'reduction_ratio'
                r_ratio_search_infos = self._get_search_option_infos(attn_search_infos=self.attention, 
                                                                     attn_type=attn_type, 
                                                                     search_option_name=r_ratio)

                intra_ops_of_this_attn_type = ChannelAttention0_Search(block_id=self.block_id, 
                                                                       channel_output=self.numChannel_InputFeature, 
                                                                       reduction_ratio_search_infos=r_ratio_search_infos,
                                                                       name=self.name)
            
            elif attn_type == 'SpatialAttention0':
                # support search options: 'kernel_size' (list of integer)
                # default static options: filters=1, stride=1, activation='softplus', epsilon=0.1

                # 'kernel_size' search infos
                k_size = 'kernel_size'
                k_size_search_infos = self._get_search_option_infos(attn_search_infos=self.attention, 
                                                                    attn_type=attn_type, 
                                                                    search_option_name=k_size)

                intra_ops_of_this_attn_type = SpatialAttention0_Search(block_id=self.block_id, 
                                                                       kernel_size_search_infos=k_size_search_infos, 
                                                                       name=self.name)
            
            else:
                raise ValueError('Unknown attention type \'{}\' for searching'.format(attn_type))

            self.intra_ops_of_attn_types.append(intra_ops_of_this_attn_type)
            

    def call(self, x, alphas_inter_attn_weights, alphas_intra_attn_weights, training=False):
        inp = x
        
        out_intra_ops_of_attn_types = list()
        
        for intra_ops_of_this_attn_type_id, intra_ops_of_this_attn_type in enumerate(self.intra_ops_of_attn_types):
            out_intra_ops_of_this_attn_type = intra_ops_of_this_attn_type(inp, alphas_intra_attn_weights)
            out_intra_ops_of_this_attn_type = out_intra_ops_of_this_attn_type * alphas_inter_attn_weights[self.block_id, intra_ops_of_this_attn_type_id]
            
            out_intra_ops_of_attn_types.append(out_intra_ops_of_this_attn_type)
        
        mix_out_intra_ops_of_attn_types = tf.add_n(out_intra_ops_of_attn_types)

        return mix_out_intra_ops_of_attn_types

    
    def _get_search_option_infos(self, attn_search_infos, attn_type, search_option_name):
        search_option_values = attn_search_infos[label_of_intra_search_infos][label_of_attn_type_search_info][attn_type][label_of_value_of_search_option][search_option_name] # list returned
        search_option_start_weight_idx = attn_search_infos[label_of_intra_search_infos][label_of_attn_type_search_info][attn_type][label_of_start_weight_idx_of_search_option][search_option_name] # integer returned
                
        search_option_infos = {label_of_value_of_search_option: search_option_values,
                               label_of_start_weight_idx_of_search_option: search_option_start_weight_idx}

        return search_option_infos