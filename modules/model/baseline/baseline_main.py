from absl import logging

from modules.model.baseline.resnet import (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

def genModelBranch(branch_id, branch_name, input_h_size, input_w_size, input_num_channel, model_name, num_layer, pretrained_weight):
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
                                 branch_model_name=f'branch_{branch_id}_{branch_name}')
    else:
        raise ValueError('Unknown model \'{}\''.format(model_name))

    # show detail of model
    logging.info('-' * 30)
    logging.info('Branch {}: {}'.format(branch_id, branch_name))
    #model_branch.summary(print_fn=lambda x: logging.info(x))
    model_branch.summary()

    return model_branch