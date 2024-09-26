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


def check_and_get_model_attention_configuration(num_branch, model_attn_cfg):
    Branches_attn_existence = model_attn_cfg['attention_existence']  # dictionary returned
    if len(Branches_attn_existence) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of attention existence list \'{}\' for branches'.format(num_branch, len(Branches_attn_existence)))
    Branches_attn_config = model_attn_cfg['attention_config_of_branches'] # dictionary returned
    if len(Branches_attn_config) != num_branch:
        raise ValueError('Number of branch \'{}\' not equal to number of attention configuration list \'{}\' for branches'.format(num_branch, len(Branches_attn_config)))
    
    return Branches_attn_existence, Branches_attn_config