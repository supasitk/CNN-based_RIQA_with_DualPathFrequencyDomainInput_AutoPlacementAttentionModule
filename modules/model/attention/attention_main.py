from modules.labels import *
from modules.model.attention.channel_attention.se import (ChannelAttention0)
from modules.model.attention.channel_attention.CBAM_ChannelAttention import (ChannelAttention1)
from modules.model.attention.channel_attention.fca import (ChannelAttention2)
from modules.model.attention.channel_attention.DepthwiseMultiFreqChannelAttention import (ChannelAttention3_1)
from modules.model.attention.spatial_attention.spatialAttention0 import (SpatialAttention0)
from modules.model.attention.spatial_attention.CBAM_SpatialAttention import (SpatialAttention1)
from modules.model.attention.channel_spatial_attention.cbam import (ChannelSpatialAttention0)


def genAttnModule(block_id, numChannel_InputFeature, attention, name=None, seed=None):
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
                                           seed=seed,
                                           name=name)

        elif attn_type == 'ChannelAttention1':
            # attn_setting of 'ChannelAttention1' -> list [idx 0:spatial_pooling_methods, 1:reduction_ratio, 2:use_bias, 3:activation_dense1, 4:activation_dense2]
            spatial_pool_methods = attn_setting[0]
            r_ratio = attn_setting[1]
            use_bias = attn_setting[2]
            act_d1 = attn_setting[3]
            act_d2 = attn_setting[4]
            
            attn_block = ChannelAttention1(channel_output=numChannel_InputFeature,
                                           spatial_pooling_methods=spatial_pool_methods,
                                           reduction_ratio=r_ratio,
                                           use_bias=use_bias, 
                                           activation_dense1=act_d1, 
                                           activation_dense2=act_d2,
                                           seed=seed,
                                           name=name)
        
        elif attn_type == 'ChannelAttention2':
            # attn_setting of 'ChannelAttention2' -> list [idx 0:freq_select_method, 1:num_freq, 2:reduction_ratio, 3:activation_dense1, 4:activation_dense2, 5:use_bias]
            freq_select_method = attn_setting[0]
            num_freq = attn_setting[1]
            r_ratio = attn_setting[2]
            act_d1 = attn_setting[3]
            act_d2 = attn_setting[4]
            use_bias = attn_setting[5]
            
            attn_block = ChannelAttention2(freq_select_method=freq_select_method,
                                           num_freq=num_freq,
                                           reduction_ratio=r_ratio, 
                                           activation_dense1=act_d1, 
                                           activation_dense2=act_d2,
                                           use_bias=use_bias,
                                           seed=seed,
                                           name=name)
            
        elif attn_type == 'ChannelAttention3_1':
            # attn_setting of 'ChannelAttention3_1' -> list [idx 0:dctSpaPool_freq_select_method, 1:dctSpaPool_num_freq, 2:multiFreqReduction_method, 3:mlp_reduction_ratio, 4:mlp_use_bias, 5:mlp_activation_dense1, 6:mlp_activation_dense2]
            dctSpaPool_freq_select_method = attn_setting[0]
            dctSpaPool_num_freq = attn_setting[1]
            multiFreqReduction_method = attn_setting[2]
            mlp_r_ratio = attn_setting[3]
            mlp_use_bias = attn_setting[4]
            mlp_act_d1 = attn_setting[5]
            mlp_act_d2 = attn_setting[6]
            
            attn_block = ChannelAttention3_1(dctSpaPool_freq_select_method=dctSpaPool_freq_select_method,
                                             dctSpaPool_num_freq=dctSpaPool_num_freq,
                                             multiFreqReduction_method=multiFreqReduction_method,
                                             mlp_reduction_ratio=mlp_r_ratio,
                                             mlp_use_bias=mlp_use_bias,
                                             mlp_activation_dense1=mlp_act_d1, 
                                             mlp_activation_dense2=mlp_act_d2,
                                             seed=seed,
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
                                           seed=seed,
                                           name=name)
            
        elif attn_type == 'SpatialAttention1':
            # attn_setting of 'SpatialAttention1' -> list [idx 0:channel_pooling_methods, 1:filter, 2:kernel_size, 3:stride, 4:activation, 5:use_bias, 6:use_bn, 7:use_relu]
            channel_pooling_methods = attn_setting[0]
            num_filter = attn_setting[1]
            kernel_size = attn_setting[2]
            stride = attn_setting[3]
            activation = attn_setting[4]
            use_bias = attn_setting[5]
            use_bn = attn_setting[6]
            use_relu = attn_setting[7]

            attn_block = SpatialAttention1(channel_pooling_methods=channel_pooling_methods, 
                                           filters=num_filter, 
                                           kernel_size=kernel_size, 
                                           strides=stride, 
                                           activation=activation, 
                                           use_bias=use_bias,
                                           use_bn=use_bn,
                                           use_relu=use_relu,
                                           seed=seed, 
                                           name=name)
    
        elif attn_type == 'ChannelSpatialAttention0':
            # attn_setting of 'ChannelSpatialAttention0' -> list [idx 0:cam_spatial_pooling_methods, 1:cam_reduction_ratio, 2:cam_use_bias, 3:cam_activation_dense1, 4:cam_activation_dense2, 
            #                                                         5:sam_channel_pooling_methods, 6:sam_filter, 7:sam_kernel_size, 8:sam_stride, 9:sam_activation, 10:sam_use_bias, 11:sam_use_bn, 12:sam_use_relu]
            cam_spatial_pool_methods = attn_setting[0]
            cam_r_ratio = attn_setting[1]
            cam_use_bias = attn_setting[2]
            cam_act_d1 = attn_setting[3]
            cam_act_d2 = attn_setting[4]

            sam_channel_pooling_methods = attn_setting[5]
            sam_num_filter = attn_setting[6]
            sam_kernel_size = attn_setting[7]
            sam_stride = attn_setting[8]
            sam_activation = attn_setting[9]
            sam_use_bias = attn_setting[10]
            sam_use_bn = attn_setting[11]
            sam_use_relu = attn_setting[12]

            attn_block = ChannelSpatialAttention0(cam_channel_output=numChannel_InputFeature, 
                                                  cam_spatial_pooling_methods=cam_spatial_pool_methods, 
                                                  cam_reduction_ratio=cam_r_ratio, 
                                                  cam_use_bias=cam_use_bias, 
                                                  cam_activation_dense1=cam_act_d1, 
                                                  cam_activation_dense2=cam_act_d2,
                                                  sam_channel_pooling_methods=sam_channel_pooling_methods, 
                                                  sam_filters=sam_num_filter, 
                                                  sam_kernel_size=sam_kernel_size, 
                                                  sam_strides=sam_stride, 
                                                  sam_activation=sam_activation, 
                                                  sam_use_bias=sam_use_bias, 
                                                  sam_use_bn=sam_use_bn, 
                                                  sam_use_relu=sam_use_relu,
                                                  seed=seed, 
                                                  name=name)

        else:
            raise ValueError('Unknown attention type \'{}\''.format(attn_type))
        
    except ValueError:
        pass

    return attn_block 