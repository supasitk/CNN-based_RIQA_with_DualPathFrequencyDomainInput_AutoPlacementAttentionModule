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

    # create tensor for calculating mean and variance of each channel of dataset
    stat_num_channel = cfg['stat_infos']['num_channel']
    total_mean = tf.zeros([stat_num_channel], tf.dtypes.float32)
    total_var = tf.zeros([stat_num_channel], tf.dtypes.float32)

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
    elif ds_transform_type == 'RGB_2resize':
        resized_img_dim = ds_transform_configs['resized_img_dim']
        rescale_value_range = ds_transform_configs['rescale_value_range']
        img_transform = cv_transform.Compose([
                            cv_transform.Resize(resized_img_dim[0]),
                            cv_transform.RescaleValue(rescale_value_range),
                            cv_transform.Resize(resized_img_dim[1])
                        ])
    elif ds_transform_type == 'DCT':
        resized_img_dim = ds_transform_configs['resized_img_dim']
        rescale_value_range = ds_transform_configs['rescale_value_range']
        color_space_conversion = ds_transform_configs['color_space_conversion']
        dct_transformation = ds_transform_configs['dct_transformation']
        img_transform = cv_transform.Compose([
                            cv_transform.Resize(resized_img_dim),
                            cv_transform.RescaleValue(rescale_value_range),
                            cv_transform.ColorSpaceConversion(color_space_conversion),
                            cv_transform.ExpandDims(axis=0),    # add batch=1 (axis=0) dimension to tensor (input: [H, W, C] -> output: [B, H, W, C])
                            cv_transform.DepthwiseDCT2d(dct_transformation),
                            cv_transform.SqueezeDims(axis=0)    # remove batch=1 (axis=0) dimension from tensor (input: [B, H, W, C] -> output: [H, W, C])
                        ])
    else:
       raise ValueError

    # calculate mean and variance of each channel of dataset (ref. https://github.com/keras-team/keras/blob/v3.3.3/keras/src/layers/preprocessing/normalization.py#L245-L277)
    # -- get infos
    ds_infos = cfg['dataset_infos']
    ds_dir = ds_infos['directory']
    # -- accumulating value
    img_names = os.listdir(ds_dir)
    total_num_img = len(img_names)
    total_count = 0
    for img_name in img_names:
        batch_count = 1
        total_count += batch_count
        print('Processing {} of {} images ({}) ...'.format(total_count, total_num_img, img_name))
        img = tf.io.read_file(os.path.join(ds_dir, img_name))
        # decode image (default type is 'RGB') to tensor with type uint8 having values in the range [0, 255]
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)
        img = img_transform(img)

        batch_mean, batch_var = tf.nn.moments(img, axes=[0, 1])
        batch_weight = float(batch_count) / total_count
        existing_weight = 1.0 - batch_weight
        new_total_mean = (
            total_mean * existing_weight + batch_mean * batch_weight
        )
        # The variance is computed using the lack-of-fit sum of squares formula (see https://en.wikipedia.org/wiki/Lack-of-fit_sum_of_squares).
        total_var = (
            total_var + (total_mean - new_total_mean) ** 2
        ) * existing_weight + (
            batch_var + (batch_mean - new_total_mean) ** 2
        ) * batch_weight
        total_mean = new_total_mean
        
    # show and write mean and variance to file
    # -- show
    print('mean = {}'.format(np.array2string(total_mean.numpy(), separator=', ')))
    print('var = {}'.format(np.array2string(total_var.numpy(), separator=', ')))
    # -- write to file
    log_file = os.path.join(cfg['log_dir'], cfg['config_summary']) + '.log'
    f = open(log_file, 'w')
    f.write(f"-----------------------------------------------------\n")
    f.write(f"Configuration summary: {cfg['config_summary']}\n")
    f.write(f"Dataset information: {cfg['dataset_infos']}\n")
    f.write(f"Statistic information: {cfg['stat_infos']}\n")
    f.write(f"Dataset transformation: {cfg['dataset_transforms']}\n")
    f.write(f"-----------------------------------------------------\n")
    f.write(f"\nmean = {np.array2string(total_mean.numpy(), separator=', ')}\n")
    f.write(f"\nvar = {np.array2string(total_var.numpy(), separator=', ')}\n")
    f.close()


if __name__ == '__main__':
    app.run(main)