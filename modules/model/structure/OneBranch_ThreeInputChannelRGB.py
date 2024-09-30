from absl import logging
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Flatten, Dropout, Dense)

from modules.labels import *
from modules.model.structure.utils import (check_and_get_model_configuration, check_and_get_model_static_attention_configuration)
from modules.model.baseline.baseline_main import (genModelBranch)
from modules.model.layer.CustomLayer import (BranchInput, SplitSoftmax)


def genModel(cfg, ds_cfg, model_cfg, model_attn_cfg, attn_search=False, attn_search_infos=None):
    # ------------------ Model Configuration ------------------ #
    input_h_size, input_w_size, input_num_channel, num_branch, Branches_idx_input_ch, Branches_name, Branches_model, Branches_numLayer, Branches_pretrainedWeight, dropout_rate, num_class = check_and_get_model_configuration(cfg, ds_cfg, model_cfg)

    # ------------------ Model Attention Configuration ------------------ #
    if attn_search == False: # no searching configuration of attention module, so getting configuration of attention module defined
        Branches_static_attn_existence, Branches_static_attn_config = check_and_get_model_static_attention_configuration(num_branch, model_attn_cfg)
    else:
        Branches_static_attn_existence = None 
        Branches_static_attn_config = None

    # ------------------ Input Layer for Data ------------------ #
    inputs = Input([input_h_size, input_w_size, input_num_channel], name='input_image')
    
    # ------ Input Layer for Attention Searching Parameter ----- #
    if attn_search == True: # searching configuration of attention modules
        # Input layer for weight of inter- and intra- attention types searching
        # note that both alphas_inter_attn and alphas_intra_attn are Input of 3-dimensional weight parameter, 
        # while defining shape of tf.keras.layers.Input by user does not need to include the batch size, 
        # (tensorflow will automatically add extra "None" dimension for the batch size being the first dimension of Input layer and then following dimensions are as the shape defined by user),
        # so defining this 'Input([None, None])' will be finally extended to 'Input([None, None, None])' (3 dimension) by tensorflow runtime,
        # however, this 1st dimension of both alphas_inter_attn and alphas_intra_attn used for set of weight parameter for each branch instead of the batch size in normal case
        alphas_inter_attn = Input([None, None], name='alpha_inter_attn')
        alphas_intra_attn = Input([None, None], name='alpha_intra_attn')

    # ------------------ Feature Extraction ------------------ #
    # 1st branch
    FirstBranch_key = 1
    FirstBranch_idx_input_ch = Branches_idx_input_ch[FirstBranch_key] # list returned 
    FirstBranch_inputs = BranchInput(Branch_idx_input_ch=FirstBranch_idx_input_ch, 
                                     name='branch_1_input')(inputs)
    # -- attention
    if attn_search == False:
        FirstBranch_attn_static_infos = {label_of_attn_existence: Branches_static_attn_existence[FirstBranch_key],
                                         label_of_attn_config: Branches_static_attn_config[FirstBranch_key]}
        FirstBranch_inter_attn_search_infos = None
        FirstBranch_intra_attn_search_infos = None
    else:
        FirstBranch_attn_static_infos = None
        # attention searching information
        FirstBranch_inter_attn_search_infos = attn_search_infos[label_of_inter][FirstBranch_key]
        FirstBranch_intra_attn_search_infos = attn_search_infos[label_of_intra][FirstBranch_key]
        # -- inter-attention
        FirstBranch_inter_num_attn_block = FirstBranch_inter_attn_search_infos[label_of_num_of_attn_block]
        FirstBranch_inter_size_splits = FirstBranch_inter_attn_search_infos[label_of_list_of_split_weights]
        FirstBranch_alphas_inter_attn_weights = SplitSoftmax(num_blocks=FirstBranch_inter_num_attn_block, 
                                                             num_ops=sum(FirstBranch_inter_size_splits), 
                                                             size_splits=FirstBranch_inter_size_splits,
                                                             name='branch_1_alpha_inter_attention_weights_SplitSoftmax')(alphas_inter_attn[FirstBranch_key-1]) # indices of 1st dimension (its size = num_branch) of alphas_inter_attn start from 0 to num_branch-1, so NthBranch_key need to be subtracted with 1
        # -- intra-attention
        FirstBranch_intra_num_attn_block = FirstBranch_intra_attn_search_infos[label_of_num_of_attn_block]
        FirstBranch_intra_size_splits = FirstBranch_intra_attn_search_infos[label_of_list_of_split_weights]
        FirstBranch_alphas_intra_attn_weights = SplitSoftmax(num_blocks=FirstBranch_intra_num_attn_block, 
                                                             num_ops=sum(FirstBranch_intra_size_splits), 
                                                             size_splits=FirstBranch_intra_size_splits,
                                                             name='branch_1_alpha_intra_attention_weights_SplitSoftmax')(alphas_intra_attn[FirstBranch_key-1]) # indices of 1st dimension (its size = num_branch) of alphas_intra_attn start from 0 to num_branch-1, so NthBranch_key need to be subtracted with 1
    
    FirstBranch_model = genModelBranch(branch_id=FirstBranch_key, 
                                       branch_name=Branches_name[FirstBranch_key],
                                       input_h_size=input_h_size,
                                       input_w_size=input_w_size,
                                       input_num_channel=len(FirstBranch_idx_input_ch), 
                                       model_name=Branches_model[FirstBranch_key], 
                                       num_layer=Branches_numLayer[FirstBranch_key],
                                       pretrained_weight=Branches_pretrainedWeight[FirstBranch_key],
                                       attn_search=attn_search,
                                       attn_static_infos=FirstBranch_attn_static_infos,
                                       attn_inter_search_infos=FirstBranch_inter_attn_search_infos,
                                       attn_intra_search_infos=FirstBranch_intra_attn_search_infos)
    FirstBranch_features = FirstBranch_model(FirstBranch_inputs)
    
    # ------------------ Feature Aggregration ------------------#
    all_features = GlobalAveragePooling2D()(FirstBranch_features)
    all_features = Flatten()(all_features)
    all_features = Dropout(dropout_rate)(all_features)
    logits = Dense(num_class, name='logits')(all_features)

    if attn_search == False:
        return Model(inputs, logits, name=cfg['model_structure'])
    else:
        return Model((inputs, alphas_inter_attn, alphas_intra_attn), logits, name="SearchAttn_" + cfg['model_structure'])