from absl import logging
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Lambda, Concatenate, GlobalAveragePooling2D, Flatten, Dropout, Dense)

from modules.labels import *
from modules.initializers import *
from modules.model.structure.utils import (check_and_get_model_configuration, check_and_get_model_static_attention_configuration)
from modules.model.baseline.baseline_main import (genModelBranch)
from modules.model.layer.CustomLayer import (BranchInput, SplitSoftmax, ImageResize, ImageRGBtoYUV, Depthwise_DCT2d)


def genModel(cfg, ds_cfg, model_cfg, model_attn_cfg, attn_search=False, attn_search_infos=None, seed=None):
    # ------------------ Model Configuration ------------------ #
    input_h_size, input_w_size, input_num_channel, num_branch, Branches_idx_input_ch, Branches_resized_input_dim, Branches_input_color_space_conversion, Branches_input_dct_transform, Branches_NN_configs, feature_aggregration, optional_dense_layers, num_class = check_and_get_model_configuration(cfg, ds_cfg, model_cfg)

    # ------------------ Model Attention Configuration ------------------ #
    if attn_search == False: # no searching configuration of attention module, so getting configuration of attention module defined
        Branches_static_attn_existence, Branches_static_attn_config = check_and_get_model_static_attention_configuration(num_branch, model_attn_cfg)
    else:
        Branches_static_attn_existence = None 
        Branches_static_attn_config = None

    # ------------------ Input Layer for Data ------------------ #
    assert input_h_size == input_w_size
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

    # ------------------ Feature Extraction (for Each Branch) ------------------ #
    list_feature_extraction_for_branches = list()
    for branch_id in range(num_branch):
        ThisBranch_key = branch_id + 1
        ThisBranch_idx_input_ch = Branches_idx_input_ch[ThisBranch_key] # list returned 
        ThisBranch_inputs = BranchInput(Branch_idx_input_ch=ThisBranch_idx_input_ch, 
                                        name=f'branch_{ThisBranch_key}_input')(inputs)
        ThisBranch_input_c_size = len(ThisBranch_idx_input_ch) 
        # -- resize
        ThisBranch_resized_input_dim = Branches_resized_input_dim[ThisBranch_key]
        ThisBranch_input_h_size, ThisBranch_input_w_size = ThisBranch_resized_input_dim, ThisBranch_resized_input_dim
        if ThisBranch_resized_input_dim != input_h_size:
            ThisBranch_inputs = ImageResize(new_height=ThisBranch_resized_input_dim, 
                                            new_width=ThisBranch_resized_input_dim, 
                                            name=f'branch_{ThisBranch_key}_resize_input')(ThisBranch_inputs)
        # -- color space conversion
        ThisBranch_input_color_space_conversion = Branches_input_color_space_conversion[ThisBranch_key]
        if ThisBranch_input_color_space_conversion is None:
            pass
        elif ThisBranch_input_color_space_conversion == 'YUV':
            ThisBranch_inputs = ImageRGBtoYUV(name=f'branch_{ThisBranch_key}_input_color_space_conversion_RGBtoYUV')(ThisBranch_inputs)
        else:
            raise ValueError('Unknown \'{}\' color space conversion of input for branch of model'.format(ThisBranch_input_color_space_conversion))
        # -- DCT transformation
        ThisBranch_input_dct_transform = Branches_input_dct_transform[ThisBranch_key] # dict returned
        if ThisBranch_input_dct_transform['transform']:
            ThisBranch_input_dct_transform_infos = ThisBranch_input_dct_transform['transform_infos'] # dict returned
            ThisBranch_input_dct_transform_type = ThisBranch_input_dct_transform_infos['type']
            if ThisBranch_input_dct_transform_type == 'Depthwise_DCT2d':
                tile_size_x = ThisBranch_input_dct_transform_infos['tile_size_x']
                tile_size_y = ThisBranch_input_dct_transform_infos['tile_size_y']
                freq_component_set = ThisBranch_input_dct_transform_infos['freq_component_set']
                freq_component_selection_of_each_channel = ThisBranch_input_dct_transform_infos['freq_component_selection_of_each_channel']
                num_selected_freq_component_of_each_channel = ThisBranch_input_dct_transform_infos['num_selected_freq_component_of_each_channel']
                
                ThisBranch_inputs = Depthwise_DCT2d(tile_size_x=tile_size_x, 
                                                    tile_size_y=tile_size_y,
                                                    freq_component_set=freq_component_set, 
                                                    freq_component_selection_of_each_channel=freq_component_selection_of_each_channel, 
                                                    num_selected_freq_component_of_each_channel=num_selected_freq_component_of_each_channel, 
                                                    name=f'branch_{ThisBranch_key}_Depthwise_DCT2d')(ThisBranch_inputs)

                ThisBranch_input_h_size = ThisBranch_input_h_size // tile_size_x
                ThisBranch_input_w_size = ThisBranch_input_w_size // tile_size_y
                ThisBranch_input_c_size = sum(num_selected_freq_component_of_each_channel)
            else:
                raise ValueError('Unknown \'{}\' DCT transform type of input for branch of model'.format(ThisBranch_input_dct_transform_type))    
        # -- attention
        if attn_search == False:
            ThisBranch_attn_static_infos = {label_of_attn_existence: Branches_static_attn_existence[ThisBranch_key],
                                            label_of_attn_config: Branches_static_attn_config[ThisBranch_key]}
            ThisBranch_inter_attn_search_infos = None
            ThisBranch_intra_attn_search_infos = None
        else:
            ThisBranch_attn_static_infos = None
            # attention searching information
            ThisBranch_inter_attn_search_infos = attn_search_infos[label_of_inter][ThisBranch_key]
            ThisBranch_intra_attn_search_infos = attn_search_infos[label_of_intra][ThisBranch_key]
            # -- inter-attention
            ThisBranch_inter_num_attn_block = ThisBranch_inter_attn_search_infos[label_of_num_of_attn_block]
            ThisBranch_inter_size_splits = ThisBranch_inter_attn_search_infos[label_of_list_of_split_weights]
            ThisBranch_alphas_inter_attn_weights = SplitSoftmax(num_blocks=ThisBranch_inter_num_attn_block, 
                                                                num_ops=sum(ThisBranch_inter_size_splits), 
                                                                size_splits=ThisBranch_inter_size_splits,
                                                                name=f'branch_{ThisBranch_key}_alpha_inter_attention_weights_SplitSoftmax')(alphas_inter_attn[ThisBranch_key-1]) # indices of 1st dimension (its size = num_branch) of alphas_inter_attn start from 0 to num_branch-1, so NthBranch_key need to be subtracted with 1
            # -- intra-attention
            ThisBranch_intra_num_attn_block = ThisBranch_intra_attn_search_infos[label_of_num_of_attn_block]
            ThisBranch_intra_size_splits = ThisBranch_intra_attn_search_infos[label_of_list_of_split_weights]
            ThisBranch_alphas_intra_attn_weights = SplitSoftmax(num_blocks=ThisBranch_intra_num_attn_block, 
                                                                num_ops=sum(ThisBranch_intra_size_splits), 
                                                                size_splits=ThisBranch_intra_size_splits,
                                                                name=f'branch_{ThisBranch_key}_alpha_intra_attention_weights_SplitSoftmax')(alphas_intra_attn[ThisBranch_key-1]) # indices of 1st dimension (its size = num_branch) of alphas_intra_attn start from 0 to num_branch-1, so NthBranch_key need to be subtracted with 1
        
        ThisBranch_input_shape_of_NN = Branches_NN_configs[ThisBranch_key]['input_shape'] # dictionary returned
        assert (ThisBranch_input_h_size == ThisBranch_input_shape_of_NN['H']) and (ThisBranch_input_w_size == ThisBranch_input_shape_of_NN['W']) and (ThisBranch_input_c_size == ThisBranch_input_shape_of_NN['C'])
        ThisBranch_model = genModelBranch(branch_id=ThisBranch_key, 
                                          branch_NN_configs=Branches_NN_configs[ThisBranch_key],
                                          input_h_size=ThisBranch_input_h_size,
                                          input_w_size=ThisBranch_input_w_size,
                                          input_num_channel=ThisBranch_input_c_size, 
                                          attn_search=attn_search,
                                          attn_static_infos=ThisBranch_attn_static_infos,
                                          attn_inter_search_infos=ThisBranch_inter_attn_search_infos,
                                          attn_intra_search_infos=ThisBranch_intra_attn_search_infos,
                                          seed=seed)
        if attn_search == False:
            ThisBranch_features = ThisBranch_model(ThisBranch_inputs)
        else:
            ThisBranch_features = ThisBranch_model((ThisBranch_inputs, ThisBranch_alphas_inter_attn_weights, ThisBranch_alphas_intra_attn_weights))

        list_feature_extraction_for_branches.append(ThisBranch_features)

    # ------------------ Prediction ------------------- #
    #-- feature aggregration
    all_features = Concatenate(axis=-1)(list_feature_extraction_for_branches)
    all_features = GlobalAveragePooling2D()(all_features)
    all_features = Flatten()(all_features)
    if feature_aggregration['dropout_rate'] != 0.0:
        all_features = Dropout(feature_aggregration['dropout_rate'], seed=seed, name=f'feature_aggregration_dropout')(all_features)
    #-- optional dense (w/wo drouput) layer(s)
    num_optional_denses = optional_dense_layers['num_dense']
    optional_denses_infos = optional_dense_layers['dense_infos'] # dict returned
    optional_denses_units = optional_denses_infos['units'] # list returned
    optional_denses_activation = optional_denses_infos['activation'] # list returned
    optional_denses_dropout_rate = optional_denses_infos['dropout_rate'] # list returned
    assert (len(optional_denses_units) == num_optional_denses) and (len(optional_denses_activation) == num_optional_denses) and (len(optional_denses_dropout_rate) == num_optional_denses)
    for optional_dense_i in range(num_optional_denses):
        all_features = Dense(units=optional_denses_units[optional_dense_i],
                             activation=optional_denses_activation[optional_dense_i],
                             kernel_initializer=DENSE_KERNEL_INIT(seed=seed),
                             name=f'optional_dense_{optional_dense_i}')(all_features)
        if optional_denses_dropout_rate[optional_dense_i] != 0.0:
            all_features = Dropout(optional_denses_dropout_rate[optional_dense_i], seed=seed, name=f'optional_dense_{optional_dense_i}_dropout')(all_features) 
    #-- final (classification) dense layer
    logits = Dense(num_class, kernel_initializer=DENSE_KERNEL_INIT(seed=seed), name='logits')(all_features)

    if attn_search == False:
        return Model(inputs, logits, name=cfg['model_structure'])
    else:
        return Model((inputs, alphas_inter_attn, alphas_intra_attn), logits, name="SearchAttn_" + cfg['model_structure'])