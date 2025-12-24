import tensorflow as tf

from modules.model.attention.channel_attention.CBAM_ChannelAttention import (ChannelAttention1)
from modules.model.attention.spatial_attention.CBAM_SpatialAttention import (SpatialAttention1)


class ChannelSpatialAttention0(tf.keras.layers.Layer):
    """Channel-Spatial Attenton block from paper 'CBAM: Convolutional Block Attention Module [2018]'."""
    def __init__(self, 
                 cam_channel_output, cam_spatial_pooling_methods=['avg', 'max'], cam_reduction_ratio=16, cam_use_bias=True, cam_activation_dense1='relu', cam_activation_dense2='sigmoid',
                 sam_channel_pooling_methods=['avg', 'max'], sam_filters=1, sam_kernel_size=7, sam_strides=1, sam_activation='sigmoid', sam_use_bias=False, sam_use_bn=True, sam_use_relu=False, 
                 seed=None, name=None, **kwargs):
        super(ChannelSpatialAttention0, self).__init__(name=name, **kwargs)

        self.cbam = tf.keras.Sequential()
        self.cbam.add(ChannelAttention1(channel_output=cam_channel_output,
                                        spatial_pooling_methods=cam_spatial_pooling_methods,
                                        reduction_ratio=cam_reduction_ratio,
                                        use_bias=cam_use_bias,
                                        activation_dense1=cam_activation_dense1,
                                        activation_dense2=cam_activation_dense2,
                                        seed=seed,
                                        name=f'{name}_cbam_cam'))
        self.cbam.add(SpatialAttention1(channel_pooling_methods=sam_channel_pooling_methods,
                                        filters=sam_filters,
                                        kernel_size=sam_kernel_size,
                                        strides=sam_strides,
                                        activation=sam_activation,
                                        use_bias=sam_use_bias,
                                        use_bn=sam_use_bn,
                                        use_relu=sam_use_relu,
                                        seed=seed,
                                        name=f'{name}_cbam_sam'))
    
    def call(self, inputs, training=False):
        return self.cbam(inputs)