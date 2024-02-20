from absl import logging

from modules.model.baseline.ResNet import (genResNet)


def genModelBranch(branch_name, input_h_size, input_w_size, input_num_channel, model_name, num_layer, pretrained_weight, weights_decay):
    if model_name == 'ResNet':
        model_branch = genResNet(input_h_size=input_h_size,
                                 input_w_size=input_w_size,
                                 input_num_channel=input_num_channel, 
                                 num_layer=num_layer, 
                                 pretrained_weight=pretrained_weight, 
                                 weights_decay=weights_decay,
                                 name=branch_name)
    else:
        raise ValueError('Unknown model \'{}\''.format(model_name))

    return model_branch