from absl import logging
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Flatten, Dropout, Dense)

from modules.labels import *
from modules.model.structure.utils import (check_and_get_model_configuration, check_and_get_model_static_attention_configuration)
from modules.model.baseline.baseline_main import (genModelBranch)
from modules.model.layer.CustomLayer import (BranchInput)


def genModel(cfg, ds_cfg, model_cfg, model_attn_cfg, attn_search=False, attn_search_infos=None):
    # ------------------ Model Configuration ------------------ #
    input_h_size, input_w_size, input_num_channel, num_branch, Branches_idx_input_ch, Branches_name, Branches_model, Branches_numLayer, Branches_pretrainedWeight, dropout_rate, num_class = check_and_get_model_configuration(cfg, ds_cfg, model_cfg)

    # ------------------ Model Attention Configuration ------------------ #
    if attn_search == False: # no searching configuration of attention module, so getting configuration of attention module defined
        Branches_static_attn_existence, Branches_static_attn_config = check_and_get_model_static_attention_configuration(num_branch, model_attn_cfg)

    # ------------------ Input Layer ------------------ #
    inputs = Input([input_h_size, input_w_size, input_num_channel], name='input_image')

    # ------------------ Feature Extraction ------------------ #
    # 1st branch
    FirstBranch_key = 1
    FirstBranch_idx_input_ch = Branches_idx_input_ch[FirstBranch_key] # list returned 
    FirstBranch_inputs = BranchInput(Branch_idx_input_ch=FirstBranch_idx_input_ch, 
                                     name='branch_1_input')(inputs)
    if attn_search == False:
        attn_static_infos = {label_of_attn_existence: Branches_static_attn_existence[FirstBranch_key],
                             label_of_attn_config :Branches_static_attn_config[FirstBranch_key]}
    else:
        attn_static_infos = None
    
    FirstBranch_model = genModelBranch(branch_id=FirstBranch_key, 
                                       branch_name=Branches_name[FirstBranch_key],
                                       input_h_size=input_h_size,
                                       input_w_size=input_w_size,
                                       input_num_channel=len(FirstBranch_idx_input_ch), 
                                       model_name=Branches_model[FirstBranch_key], 
                                       num_layer=Branches_numLayer[FirstBranch_key],
                                       pretrained_weight=Branches_pretrainedWeight[FirstBranch_key],
                                       attn_search=attn_search,
                                       attn_static_infos=attn_static_infos)
    FirstBranch_features = FirstBranch_model(FirstBranch_inputs)
    
    # ------------------ Feature Aggregration ------------------#
    all_features = GlobalAveragePooling2D()(FirstBranch_features)
    all_features = Flatten()(all_features)
    all_features = Dropout(dropout_rate)(all_features)
    logits = Dense(num_class, name='logits')(all_features)

    return Model(inputs, logits, name=cfg['model_structure'])