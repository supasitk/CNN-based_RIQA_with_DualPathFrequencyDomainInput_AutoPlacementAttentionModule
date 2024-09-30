from modules.labels import *


def check_and_get_model_configuration(cfg, ds_cfg, model_cfg):
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

    return input_h_size, input_w_size, input_num_channel, num_branch, Branches_idx_input_ch, Branches_name, Branches_model, Branches_numLayer, Branches_pretrainedWeight, dropout_rate, num_class


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
    
    attn = {label_of_attn_existence: attn_exist,
            label_of_attn_block_id: attn_block_id,
            label_of_attn_type: attn_type,
            label_of_attn_setting: attn_setting}

    return attn