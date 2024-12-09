from modules.labels import *


def check_and_get_model_configuration(cfg, ds_cfg, model_cfg):
    input_h_size = ds_cfg['resized_image_dim']
    input_w_size = ds_cfg['resized_image_dim']
    input_num_channel = model_cfg['number_of_input_channel']
    num_branch = model_cfg['number_of_branch']
    Branches_idx_input_ch = model_cfg['indices_of_input_channel_for_branches'] # dictionary returned
    if len(Branches_idx_input_ch) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of index set \'{}\' of input channel for branches'.format(num_branch, len(Branches_idx_input_ch)))
    Branches_input_color_space_conversion = model_cfg['color_space_conversion_for_branches'] # dictionary returned
    if len(Branches_input_color_space_conversion) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of input color space conversion list \'{}\' for branches'.format(num_branch, len(Branches_input_color_space_conversion)))
    Branches_NN_configs = model_cfg['NN_config_of_branch'] # dictionary returned
    if len(Branches_NN_configs) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of neural network configurations list \'{}\' for branches'.format(num_branch, len(Branches_NN_configs)))
    dropout_rate = cfg['dropout_rate']
    num_class = ds_cfg['class_num']

    return input_h_size, input_w_size, input_num_channel, num_branch, Branches_idx_input_ch, Branches_input_color_space_conversion, Branches_NN_configs, dropout_rate, num_class


def check_and_get_model_static_attention_configuration(num_branch, model_attn_cfg):
    Branches_attn_existence = model_attn_cfg['attention_existence']  # dictionary returned
    if len(Branches_attn_existence) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of attention existence list \'{}\' for branches'.format(num_branch, len(Branches_attn_existence)))
    Branches_attn_config = model_attn_cfg['attention_config_of_branches'] # dictionary returned
    if len(Branches_attn_config) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of attention configuration list \'{}\' for branches'.format(num_branch, len(Branches_attn_config)))
    
    return Branches_attn_existence, Branches_attn_config


def get_static_attention_configuration_for_branch(attn_static_infos):
    attn_block_id = list()
    attn_type = list()
    attn_setting = list()

    attn_exist = attn_static_infos[label_of_attn_existence]
    if attn_exist: # using attention for this model branch
        for attn_config_layer in attn_static_infos[label_of_attn_config]: # attn_config is a list of dictionary with key: 'block_id' (interger), 'attn_type' (string), 'attn_config' (list)
            attn_block_id.append(attn_config_layer[label_of_block_id])
            attn_type.append(attn_config_layer[label_of_attn_type])
            attn_setting.append(attn_config_layer[label_of_attn_config])
    
    attn = {label_of_attn_search: False,
            label_of_attn_existence: attn_exist,
            label_of_attn_block_id: attn_block_id,
            label_of_attn_type: attn_type,
            label_of_attn_setting: attn_setting}

    return attn


def get_search_attention_configuration_for_branch(attn_inter_search_infos, attn_intra_search_infos):
    attn = {label_of_attn_search: True,
            label_of_inter_search_infos: attn_inter_search_infos,
            label_of_intra_search_infos: attn_intra_search_infos}

    return attn