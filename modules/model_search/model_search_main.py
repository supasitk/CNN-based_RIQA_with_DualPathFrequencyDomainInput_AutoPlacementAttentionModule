from absl import logging
import tensorflow as tf

from modules.model_search.structure_search import (Search_OneBranch_ThreeInputChannelRGB)


class SearchNNmodel(object):
    """Searching Neural Network Model"""
    def __init__(self, cfg, ds_cfg, model_cfg, model_attn_search_cfg, name='SearchNNmodel'):
        logging.info(f'Building {name}...')

        cfg_model_structure = cfg['model_structure']

        # if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
        #     self.model = Search_OneBranch_ThreeInputChannelRGB.genModel(cfg, ds_cfg, model_cfg, model_attn_search_cfg)
        # else:
        #     raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))