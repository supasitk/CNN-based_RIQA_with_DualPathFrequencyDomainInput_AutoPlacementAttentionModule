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

        # generate weight parameter for searching which attention type module is the optimized option (selecting the optimized attention type among all attention types searched) for each block attached of each branch on the model structure,
        # the dimension of this weight parameter is (number_of_branch, MAX(number_of_attention_block[each_branch]), MAX(number_of_attention_type[each_branch])),
        # note that the number of this weight parameter might be created redundant (excess) that some of them might be not used, the reason is to simply create tensor with same dimension for every branches.  
        self.inter_attn_search_parameters = self._initialize_inter_attn_search_parameter()

        # build model
        logging.info(f'Building {self.name}...')
        cfg_model_structure = self.cfg['model_structure']
        # if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
        #     self.model = Search_OneBranch_ThreeInputChannelRGB.genModel(self.cfg, self.ds_cfg, self.model_cfg, self.model_attn_search_cfg)
        # else:
        #     raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))
    

    def _initialize_inter_attn_search_parameter(self):
        num_of_branch = self.model_cfg['number_of_branch']
        list_of_num_of_attn_block_for_each_branch = list()
        list_of_num_of_attn_type_for_each_branch = list()
        for branch_id in range(1, num_of_branch+1):
            list_of_num_of_attn_block_for_each_branch.append(self.model_attn_search_cfg['number_of_attention_block'][branch_id])
            list_of_num_of_attn_type_for_each_branch.append(self.model_attn_search_cfg['number_of_attention_type'][branch_id])
        max_num_of_attn_block_for_any_branch = max(list_of_num_of_attn_block_for_each_branch)
        max_num_of_attn_type_for_any_branch = max(list_of_num_of_attn_type_for_each_branch)

        w_init = tf.random_normal_initializer()
        self.inter_attn_alphas = tf.Variable(initial_value=1e-3 * w_init(shape=[num_of_branch, max_num_of_attn_block_for_any_branch, max_num_of_attn_type_for_any_branch], dtype='float32'),
                                             trainable=True, name='inter_attn_alphas')
        return [self.inter_attn_alphas]
