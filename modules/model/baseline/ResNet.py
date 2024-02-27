from absl import logging
import tensorflow as tf

from tensorflow.keras.utils import get_file
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Input, ReLU, MaxPool2D)

from modules.utils import Check_Layer_Not_Loaded_with_PretrainedWeight
from modules.operations import (kernel_init, regularizer)

# tensorflow weights path (https://storage.googleapis.com/tensorflow/)
# ImageNet 
TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/'
)
TF_IMAGENET_RESNET_SUFFIX_WEIGHTS_PATH = (
    '_weights_tf_dim_ordering_tf_kernels.h5'
)
TF_IMAGENET_RESNET_NOTOP_SUFFIX_WEIGHTS_PATH = (
    '_weights_tf_dim_ordering_tf_kernels_notop.h5'
)
TF_IMAGENET_RESNET_WEIGHTS_HASHES = {    # index 0: IncludeTop, 1: NoIncludeTop
    'resnet50': (
        '2cb95161c43110f7111970584f804107',
        '4d473c1dd8becc155b73f8504c6f6626'
    ),
    'resnet101': (
        'f1aeb4b969a6efcfb50fad2f0c20cfc5',
        '88cf7a10940856eca736dc7b7e228a21'
    ),
    'resnet152': (
        '100835be76be38e30d865e96f2aaae62',
        'ee4c566cf9a93f14d82f913c2dc6dd0c'
    ),
}


class _Basic_Residual_block(tf.keras.layers.Layer):
    """Basic Residual block"""
    expansion = 1
    BN_DECAY = 0.9
    BN_EPSILON = 1e-5

    def __init__(self, ch_out, stride, wd, name, **kwargs):
        super(_Basic_Residual_block, self).__init__(name=name, **kwargs) 
        self.ch_out = ch_out
        self.stride = stride
        self.wd = wd


    def build(self, input_shape):
        ch_in = input_shape[-1]

        self.conv1 = Conv2D(filters=self.ch_out, 
                            kernel_size=3, 
                            strides=self.stride,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.bn1 = BatchNormalization(momentum=self.BN_DECAY, 
                                      epsilon=self.BN_EPSILON)
    
        self.conv2 = Conv2D(filters=self.ch_out,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.bn2 = BatchNormalization(momentum=self.BN_DECAY,
                                      epsilon=self.BN_EPSILON)


        # add a shortcut between input and residual block and merge them with "sum"
        if self.stride != 1 or ch_in != self.expansion * self.ch_out:
            self.shortcut = Sequential([Conv2D(filters=self.expansion*self.ch_out, 
                                               kernel_size=1, 
                                               strides=self.stride, 
                                               use_bias=False, 
                                               kernel_initializer=kernel_init(), 
                                               kernel_regularizer=regularizer(self.wd)), 
                                        BatchNormalization(momentum=self.BN_DECAY,
                                                           epsilon=self.BN_EPSILON)
                                       ])
        else:
            self.shortcut = lambda x: x


    def call(self, inputs):
        out = tf.nn.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        return tf.nn.relu(out)


class _Bottleneck_Residual_block(tf.keras.layers.Layer):
    """Bottleneck Residual block"""
    expansion = 4
    BN_DECAY = 0.9
    BN_EPSILON = 1e-5

    def __init__(self, ch_out, stride, wd, name, **kwargs):
        super(_Bottleneck_Residual_block, self).__init__(name=name, **kwargs) 
        self.ch_out = ch_out
        self.stride = stride
        self.wd = wd
        

    def build(self, input_shape):
        ch_in = input_shape[-1]

        self.conv1 = Conv2D(filters=self.ch_out, 
                            kernel_size=1, 
                            strides=1,
                            use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.bn1 = BatchNormalization(momentum=self.BN_DECAY, 
                                      epsilon=self.BN_EPSILON)
    
        self.conv2 = Conv2D(filters=self.ch_out,
                            kernel_size=3,
                            strides=self.stride,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.bn2 = BatchNormalization(momentum=self.BN_DECAY,
                                      epsilon=self.BN_EPSILON)
    
        self.conv3 = Conv2D(filters=self.expansion*self.ch_out,
                            kernel_size=1,
                            strides=1,
                            use_bias=False,
                            kernel_initializer=kernel_init(),
                            kernel_regularizer=regularizer(self.wd))
        self.bn3 = BatchNormalization(momentum=self.BN_DECAY,
                                      epsilon=self.BN_EPSILON)   
       
        # add a shortcut between input and residual block and merge them with "sum"
        if self.stride != 1 or ch_in != self.expansion * self.ch_out:
            self.shortcut = Sequential([Conv2D(filters=self.expansion*self.ch_out, 
                                               kernel_size=1, 
                                               strides=self.stride, 
                                               use_bias=False, 
                                               kernel_initializer=kernel_init(), 
                                               kernel_regularizer=regularizer(self.wd)),
                                        BatchNormalization(momentum=self.BN_DECAY,
                                                           epsilon=self.BN_EPSILON)
                                       ])
        else:
            self.shortcut = lambda x: x


    def call(self, inputs):
        out = tf.nn.relu(self.bn1(self.conv1(inputs)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inputs)
        return tf.nn.relu(out)


def genResNet(input_h_size, input_w_size, input_num_channel, num_layer, pretrained_weight, weights_decay, id, name):
    # common configurations
    num_stage = 4
    ch_outs = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    if num_layer == 18:
        CellClass = _Basic_Residual_block
        num_blocks = [2, 2, 2, 2]
    elif num_layer == 34:
        CellClass = _Basic_Residual_block
        num_blocks = [3, 4, 6, 3]
    elif num_layer == 50:
        if pretrained_weight == 'imagenet':
            ModelName = 'resnet50'
            WeightName = ModelName + TF_IMAGENET_RESNET_SUFFIX_WEIGHTS_PATH
            WeightPath = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName
            WeightHash = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][0]
            WeightName_NoTop = ModelName + TF_IMAGENET_RESNET_NOTOP_SUFFIX_WEIGHTS_PATH
            WeightPath_NoTop = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName_NoTop
            WeightHash_NoTop = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][1]
        CellClass = _Bottleneck_Residual_block
        num_blocks = [3, 4, 6, 3]
    elif num_layer == 101:
        if pretrained_weight == 'imagenet':
            ModelName = 'resnet101'
            WeightName = ModelName + TF_IMAGENET_RESNET_SUFFIX_WEIGHTS_PATH
            WeightPath = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName
            WeightHash = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][0]
            WeightName_NoTop = ModelName + TF_IMAGENET_RESNET_NOTOP_SUFFIX_WEIGHTS_PATH
            WeightPath_NoTop = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName_NoTop
            WeightHash_NoTop = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][1]
        CellClass = _Bottleneck_Residual_block
        num_blocks = [3, 4, 23, 3]
    elif num_layer == 152:
        if pretrained_weight == 'imagenet':
            ModelName = 'resnet152'
            WeightName = ModelName + TF_IMAGENET_RESNET_SUFFIX_WEIGHTS_PATH
            WeightPath = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName
            WeightHash = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][0]
            WeightName_NoTop = ModelName + TF_IMAGENET_RESNET_NOTOP_SUFFIX_WEIGHTS_PATH
            WeightPath_NoTop = TF_IMAGENET_RESNET_BASE_WEIGHTS_PATH + WeightName_NoTop
            WeightHash_NoTop = TF_IMAGENET_RESNET_WEIGHTS_HASHES[ModelName][1]
        CellClass = _Bottleneck_Residual_block
        num_blocks = [3, 8, 36, 3]
    else:
        raise ValueError('Unknown number of layer (\'{}\') for Resnet'.format(num_layer))

    # define model
    inputs = Input([input_h_size, input_w_size, input_num_channel], name=f'b{id}-{name}_input')

    stem_conv = Sequential([
                    Conv2D(filters=64, kernel_size=7, strides=2, padding='same',
                           kernel_initializer=kernel_init(), 
                           kernel_regularizer=regularizer(weights_decay), use_bias=False),
                    BatchNormalization(momentum=0.9, epsilon=1e-5),
                    ReLU(),
                    MaxPool2D(pool_size=3, strides=2, padding='same')],
                    name=f'b{id}-{name}_stemConv')(inputs)

    x = stem_conv
    
    for stage in range(num_stage):
        stage_strides = [strides[stage]] + [1] * (num_blocks[stage] - 1)
        for block in range(num_blocks[stage]):
            cell = CellClass(ch_out=ch_outs[stage], stride=stage_strides[block], wd=weights_decay, name=f'b{id}-{name}_conv_s{stage}_b{block}') 
            x = cell(x)

    fea = x

    model = Model(inputs, fea, name=f'b{id}-{name}')
    # show detail of model
    logging.info('-' * 30)
    logging.info('Branch {}: {}'.format(id, name))
    model.summary(print_fn=lambda x: logging.info(x))

    # load weights
    if pretrained_weight is not None:
        # keep initial weights of the model for later checking which layers are loaded (replaced) with the pretrained weights
        initial_weights = [layer.get_weights() for layer in model.layers]

        if pretrained_weight == 'imagenet':
            loaded_weights = get_file(WeightName_NoTop,
                                      WeightPath_NoTop,
                                      cache_subdir='pretrained_weights',
                                      file_hash=WeightHash_NoTop)
            model.load_weights(loaded_weights)
        else:
            model.load_weights(pretrained_weight)

            logging.info('Pretrained weights loaded for branch {} \'{}\': {}'.format(id, name, pretrained_weight))
            # checking which layers are NOT loaded (replaced) with the pretrained weights
            Check_Layer_Not_Loaded_with_PretrainedWeight(model, initial_weights)
    else:
        logging.info('Pretrained weights loaded for branch {} \'{}\': NONE'.format(id, name))

    return model