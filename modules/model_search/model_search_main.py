from absl import logging
import tensorflow as tf

from modules.model_search.structure_search import (Search_OneBranch_ThreeInputChannelRGB)


class SearchNNmodel(object):
    """Searching Neural Network Model"""
    def __init__(self, cfg, ds_cfg, model_cfg, model_attn_search_cfg, name='SearchNNmodel'):
        self.cfg = cfg
        self.ds_cfg = ds_cfg 
        self.model_cfg = model_cfg 
        self.model_attn_search_cfg = model_attn_search_cfg 
        self.name = name

        # # generate weight parameter for searching attention module
        # self.attn_search_parameters = self._initialize_alphas()

        # build model
        logging.info(f'Building {self.name}...')
        cfg_model_structure = self.cfg['model_structure']
        # if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
        #     self.model = Search_OneBranch_ThreeInputChannelRGB.genModel(self.cfg, self.ds_cfg, self.model_cfg, self.model_attn_search_cfg)
        # else:
        #     raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))