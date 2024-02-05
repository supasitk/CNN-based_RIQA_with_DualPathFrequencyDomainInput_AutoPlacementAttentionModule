from absl import logging
import tensorflow as tf

from tensorflow.keras.layers import (Input)


def genModel(cfg, ds_cfg, model_cfg):
    # model configuration
    num_branch = model_cfg['number_of_branch']
    input_h_size = ds_cfg['resized_image_dim']
    input_w_size = ds_cfg['resized_image_dim']
    input_num_channel = model_cfg['number_of_input_channel']
    Branches_idx_input_ch = model_cfg['indices_of_input_channel_for_branches']  # dictionary returned
    if len(Branches_idx_input_ch) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of index set \'{}\' of input channel for branches'.format(num_branch, len(Branches_idx_input_ch)))
    
    # input layer
    inputs = Input([input_h_size, input_w_size, input_num_channel], name='input_image')

    # 1st branch
    FirstBranch_key = 1
    FirstBranch_idx_input_ch = Branches_idx_input_ch[FirstBranch_key] # list returned 
    FirstBranch_inputs = tf.gather(inputs, FirstBranch_idx_input_ch, axis=-1, name='1st_branch_input')

    return None