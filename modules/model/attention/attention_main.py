from modules.labels import *
from modules.model.attention.channel_attention.se import (ChannelAttention0)
from modules.model.attention.spatial_attention.spatialAttention0 import (SpatialAttention0)


def genAttnModule(block_id, numChannel_InputFeature, attention, name=None):
    attn_block = None
                                       
    try:
        # check if attention module attached to this block 
        # if 'block_id' exists on list of 'attn_block_id' returns 'index' in the list which use further to get attention configuration for this block,
        # otherwise returns ValueError
        idx = attention[label_of_attn_block_id].index(block_id)
        attn_type = attention[label_of_attn_type][idx]
        attn_setting = attention[label_of_attn_setting][idx]

        if attn_type == 'ChannelAttention0':
            # attn_setting of 'ChannelAttention0' -> list [idx 0:reduction_ratio, 1:activation_dense1, 2:activation_dense2]
            r_ratio = attn_setting[0]
            act_d1 = attn_setting[1]
            act_d2 = attn_setting[2]
            
            attn_block = ChannelAttention0(channel_output=numChannel_InputFeature,
                                           reduction_ratio=r_ratio, 
                                           activation_dense1=act_d1, 
                                           activation_dense2=act_d2,
                                           name=name)
    
        elif attn_type == 'SpatialAttention0':
            # attn_setting of 'SpatialAttention0' -> list [idx 0:filter, 1:kernel_size, 2:stride, 3:activation, 4:epsilon]
            num_filter = attn_setting[0]
            kernel_size = attn_setting[1]
            stride = attn_setting[2]
            activation = attn_setting[3]
            epsilon = attn_setting[4]
            
            attn_block = SpatialAttention0(filters=num_filter,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation=activation,
                                           epsilon=epsilon,
                                           name=name)
    
        else:
            raise ValueError('Unknown attention type \'{}\''.format(attn_type))
        
    except ValueError:
        pass

    return attn_block 