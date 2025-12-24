from absl import logging
import tensorflow as tf

from modules.model.structure import (model_structure)


def NNmodel(cfg, ds_cfg, model_cfg, model_attn_cfg, name='NNmodel', seed=None):
    """Neural Network Model"""
    cfg_model_structure = cfg['model_structure']
    
    logging.info(f'Building {name}: {cfg_model_structure} ...')
    model = model_structure.genModel(cfg, ds_cfg, model_cfg, model_attn_cfg, seed=seed)
    
    return model