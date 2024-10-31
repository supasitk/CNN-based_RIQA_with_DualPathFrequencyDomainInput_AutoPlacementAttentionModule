from absl import logging
import tensorflow as tf

from modules.model.structure import (OneBranch_ThreeInputChannelRGB)


def NNmodel(cfg, ds_cfg, model_cfg, model_attn_cfg, name='NNmodel', seed=None):
    """Neural Network Model"""
    logging.info(f'Building {name}...')

    cfg_model_structure = cfg['model_structure']

    if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
        model = OneBranch_ThreeInputChannelRGB.genModel(cfg, ds_cfg, model_cfg, model_attn_cfg, seed=seed)
    else:
        raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))

    return model