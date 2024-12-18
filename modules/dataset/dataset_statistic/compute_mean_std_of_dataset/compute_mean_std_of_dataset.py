from absl import app, flags
from absl.flags import FLAGS
import os
import numpy as np
import tensorflow as tf

from modules.utils import (load_yaml, set_memory_growth)
import modules.dataset.utils.cv_transform as cv_transform

flags.DEFINE_string('cfg_path', './modules/dataset/dataset_statistic/compute_mean_std_of_dataset/config_compute_mean_std_of_dataset.yaml', 'config file path')


def main(_):
    # load configuration
    cfg = load_yaml(FLAGS.cfg_path)

    # set log level of tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set memory growth in tensorflow
    set_memory_growth()

    # create tensor for accumulatng value of each channel of each input to finally calculate their mean and variance of dataset
    stat_num_channel = cfg['stat_infos']['num_channel']
    mean = tf.zeros([stat_num_channel], tf.dtypes.float32)
    var = tf.zeros([stat_num_channel], tf.dtypes.float32)

    # dataset transform pipeline
    # -- get infos 
    ds_transform = cfg['dataset_transforms']
    ds_transform_type = ds_transform['type']
    ds_transform_configs = ds_transform['configs']
    # -- generate pipeline
    if ds_transform_type == 'RGB':
        resized_img_dim = ds_transform_configs['resized_img_dim']
        rescale_value_range = ds_transform_configs['rescale_value_range']
        img_transform = cv_transform.Compose([
                            cv_transform.Resize(resized_img_dim),
                            cv_transform.RescaleValue(rescale_value_range)
                        ])
    elif ds_transform_type == 'Depthwise_DCT2d':
        resized_img_dim = ds_transform_configs['resized_img_dim']
        rescale_value_range = ds_transform_configs['rescale_value_range']
        color_space_conversion = ds_transform_configs['color_space_conversion']
        dct_infos = ds_transform_configs['dct_infos']
        img_transform = cv_transform.Compose([
                            cv_transform.Resize(resized_img_dim),
                            cv_transform.RescaleValue(rescale_value_range),
                            cv_transform.ColorSpaceConversion(color_space_conversion),
                            cv_transform.ExpandDims(axis=0),            # add batch=1 (axis=0) dimension to tensor (input: [H, W, C] -> output: [B, H, W, C])
                            cv_transform.DepthwiseDCT2d(dct_infos),     # receive 4D-tensor for input
                            cv_transform.SqueezeDims(axis=0)            # remove batch=1 (axis=0) dimension from tensor (input: [B, H, W, C] -> output: [H, W, C])
                        ])
    else:
       raise ValueError

    # accumulating value of each channel of each input
    # -- get infos
    ds_infos = cfg['dataset_infos']
    ds_dir = ds_infos['directory']
    # -- accumulating value
    img_names = os.listdir(ds_dir)
    total_num_img = len(img_names)
    accumulated_num_img = 0
    for img_name in img_names:
        accumulated_num_img += 1
        print('Processing {} of {} images ({}) ...'.format(accumulated_num_img, total_num_img, img_name))
        img = tf.io.read_file(os.path.join(ds_dir, img_name))
        # decode image (default type is 'RGB') to tensor with type uint8 having values in the range [0, 255]
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)
        img = img_transform(img)

        this_mean, this_var = tf.nn.moments(img, axes=[0, 1])
        mean = tf.math.add(mean, this_mean)
        var = tf.math.add(var, this_var)
    
    # calculate mean/var/std of each channel of dataset
    mean = tf.math.divide(mean, accumulated_num_img)
    var = tf.math.divide(var, accumulated_num_img)
    std = tf.math.sqrt(var)

    # show and write mean/var/std to file
    print('mean = {}'.format(np.array2string(mean.numpy(), separator=', ')))
    print('var = {}'.format(np.array2string(var.numpy(), separator=', ')))
    print('std = {}'.format(np.array2string(std.numpy(), separator=', ')))


if __name__ == '__main__':
    app.run(main)