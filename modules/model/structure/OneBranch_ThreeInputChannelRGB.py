from absl import logging
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Flatten, Dropout, Dense)

from modules.model.baseline.baseline_main import (genModelBranch)


def genModel(cfg, ds_cfg, model_cfg):
    # ------------------ Model Configuration ------------------ #
    input_h_size = ds_cfg['resized_image_dim']
    input_w_size = ds_cfg['resized_image_dim']
    input_num_channel = model_cfg['number_of_input_channel']
    num_branch = model_cfg['number_of_branch']
    Branches_idx_input_ch = model_cfg['indices_of_input_channel_for_branches'] # dictionary returned
    if len(Branches_idx_input_ch) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of index set \'{}\' of input channel for branches'.format(num_branch, len(Branches_idx_input_ch)))
    Branches_name = model_cfg['name_of_branches'] # dictionary returned
    if len(Branches_name) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of name list \'{}\' for branches'.format(num_branch, len(Branches_name)))
    Branches_model = model_cfg['model_of_branches'] # dictionary returned
    if len(Branches_model) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of model list \'{}\' for branches'.format(num_branch, len(Branches_model)))
    Branches_numLayer = model_cfg['numLayer_of_branches'] # dictionary returned
    if len(Branches_numLayer) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of layer list \'{}\' for branches'.format(num_branch, len(Branches_numLayer)))
    Branches_pretrainedWeight = model_cfg['pretrainedWeight_of_branches'] # dictionary returned
    if len(Branches_pretrainedWeight) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of pretrained-weight list \'{}\' for branches'.format(num_branch, len(Branches_pretrainedWeight)))
    dropout_rate = cfg['dropout_rate']
    num_class = ds_cfg['class_num']

    # ------------------ Input Layer ------------------ #
    inputs = Input([input_h_size, input_w_size, input_num_channel], name='input_image')

    # ------------------ Feature Extraction ------------------ #
    # 1st branch
    FirstBranch_key = 1
    FirstBranch_idx_input_ch = Branches_idx_input_ch[FirstBranch_key] # list returned 
    FirstBranch_inputs = tf.gather(inputs, FirstBranch_idx_input_ch, axis=-1, name='1st_branch_input')
    FirstBranch_model = genModelBranch(branch_id=FirstBranch_key, 
                                       branch_name=Branches_name[FirstBranch_key],
                                       input_h_size=input_h_size,
                                       input_w_size=input_w_size,
                                       input_num_channel=len(FirstBranch_idx_input_ch), 
                                       model_name=Branches_model[FirstBranch_key], 
                                       num_layer=Branches_numLayer[FirstBranch_key],
                                       pretrained_weight=Branches_pretrainedWeight[FirstBranch_key])
    FirstBranch_features = FirstBranch_model(FirstBranch_inputs)
    
    # ------------------ Feature Aggregration ------------------#
    all_features = GlobalAveragePooling2D()(FirstBranch_features)
    all_features = Flatten()(all_features)
    all_features = Dropout(dropout_rate)(all_features)
    logits = Dense(num_class, name='logits')(all_features)

    return Model(inputs, logits, name=cfg['model_structure'])