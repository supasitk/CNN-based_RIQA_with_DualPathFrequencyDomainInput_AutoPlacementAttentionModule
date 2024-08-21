from absl import logging

from modules.model.baseline.resnet import (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

def genModelBranch(branch_id, branch_name, input_h_size, input_w_size, input_num_channel, model_name, num_layer, pretrained_weight, attn_exist=False, attn_config=None):
    attn_block_id = list()
    attn_type = list()
    attn_setting = list()
    if attn_exist: # using attention for this model branch
        for attn_config_layer in attn_config: # attn_config is a list of dictionary with key: 'block_id' (interger), 'attn_type' (string), 'attn_config' (list)
            attn_block_id.append(attn_config_layer['block_id'])
            attn_type.append(attn_config_layer['attn_type'])
            attn_setting.append(attn_config_layer['attn_config'])
    attn = {'attn_exist': attn_exist,
            'attn_block_id': attn_block_id,
            'attn_type': attn_type,
            'attn_setting': attn_setting}    

    if model_name == 'ResNet':
        if num_layer == 18:
            genResNet = ResNet18
        elif num_layer == 34:
            genResNet = ResNet34
        elif num_layer == 50:
            genResNet = ResNet50
        elif num_layer == 101:
            genResNet = ResNet101
        elif num_layer == 152:
            genResNet = ResNet152
        else:
            raise ValueError('Unknown number of layer \'{}\' for model \'{}\''.format(num_layer, model_name))
        model_branch = genResNet(include_top=False,
                                 weights=pretrained_weight,
                                 input_tensor=None,
                                 input_shape=(input_h_size, input_w_size, input_num_channel),
                                 pooling=None,
                                 branch_model_name=f'branch_{branch_id}_{branch_name}',
                                 attention=attn)
    else:
        raise ValueError('Unknown model \'{}\''.format(model_name))

    # show detail of model
    logging.info('-' * 30)
    logging.info('Branch {}: {}'.format(branch_id, branch_name))
    #model_branch.summary(print_fn=lambda x: logging.info(x))
    model_branch.summary()

    return model_branch