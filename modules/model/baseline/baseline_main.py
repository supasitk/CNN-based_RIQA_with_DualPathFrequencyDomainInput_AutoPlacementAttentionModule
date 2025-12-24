from absl import logging

from modules.labels import *
from modules.model.structure.utils import (get_static_attention_configuration_for_branch, get_search_attention_configuration_for_branch)
from modules.model.baseline.resnet import (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

def genModelBranch(branch_id, branch_NN_configs, input_h_size, input_w_size, input_num_channel, 
                   attn_search=False, attn_static_infos=None, attn_inter_search_infos=None, attn_intra_search_infos=None, seed=None):
    # ------------------ Model Branch Configuration ------------------ #
    branch_name = branch_NN_configs['name']
    model_name = branch_NN_configs['model']
    
    # ------------------ Attention Configuration ------------------ #
    if attn_search == False:
        attn = get_static_attention_configuration_for_branch(attn_static_infos)
    else:
        attn = get_search_attention_configuration_for_branch(attn_inter_search_infos, attn_intra_search_infos)    

    if model_name == 'ResNet':
        # get configurations of ResNet
        num_layer = branch_NN_configs['model_infos']['numLayer']
        pretrained_weight = branch_NN_configs['model_infos']['pretrainedWeight']
        excluded_parts = branch_NN_configs['model_infos']['excluded_parts']
        included_parts = branch_NN_configs['model_infos']['included_parts']

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
                                 excluded_parts=excluded_parts,
                                 included_parts=included_parts,
                                 attention=attn,
                                 seed=seed)
    else:
        raise ValueError('Unknown model \'{}\''.format(model_name))

    # show detail of model
    logging.info('-' * 30)
    logging.info('Branch {}: {}'.format(branch_id, branch_name))
    #model_branch.summary(print_fn=lambda x: logging.info(x))
    model_branch.summary()

    return model_branch