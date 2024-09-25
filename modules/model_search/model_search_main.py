from absl import logging
import tensorflow as tf

from modules.model_search.structure_search import (Search_OneBranch_ThreeInputChannelRGB)
from modules.utils import (write_info_and_weight_parameter_for_attention_searching)
from modules.labels import *


class SearchNNmodel(object):
    """Searching Neural Network Model"""
    def __init__(self, cfg, ds_cfg, model_cfg, model_attn_search_cfg, name='SearchNNmodel'):
        self.cfg = cfg
        self.ds_cfg = ds_cfg 
        self.model_cfg = model_cfg 
        self.model_attn_search_cfg = model_attn_search_cfg 
        self.name = name

        # generate weight parameter and information for INTER-attention_types searching to find which attention type module is the optimized option (selecting the optimized attention type among all attention types searched) for each block attached of each branch on the model structure,
        # the dimension of this weight parameter is (number_of_branch, MAX(number_of_attention_block[each_branch]), MAX(number_of_attention_type[each_branch])), note that the number of this weight parameter might be created redundant (excess) that some of them might be not used, the reason is to simply create tensor with same dimension for every branches.
        # the information will be used for being detail when build and call attention type module for searching  
        self.inter_attn_search_parameters, self.inter_attn_search_infos = self._initialize_inter_attn_search_parameter_and_info()
        write_info_and_weight_parameter_for_attention_searching('INTER', self.inter_attn_search_infos, self.inter_attn_search_parameters)

        # build model
        logging.info(f'Building {self.name}...')
        cfg_model_structure = self.cfg['model_structure']
        # if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
        #     self.model = Search_OneBranch_ThreeInputChannelRGB.genModel(self.cfg, self.ds_cfg, self.model_cfg, self.model_attn_search_cfg)
        # else:
        #     raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))
    

    def _initialize_inter_attn_search_parameter_and_info(self):
        num_of_branch = self.model_cfg['number_of_branch']

        # information of inter-attention_types searching (dict)
        # for each branch_id: (dict)
        #   {
        #    'number_of_attention_block': integer,
        #    'number_of_attention_type': integer,
        #    'list_of_attention_type': list,
        #    'attention_type_search_info': dict {
        #                                        attention_type_name (string): dict {'weight_index': integer} (for attention type #1),
        #                                        ............................................................ (for attention type #N)                                         
        #                                       }      
        #   }        
        self.inter_attn_search_infos = dict()

        list_of_num_of_attn_block_for_each_branch = list()
        list_of_num_of_attn_type_for_each_branch = list()

        for branch_id in range(1, num_of_branch+1):
            # index of weight parameter assigned for each attention type 
            weight_idx = 0

            num_of_attn_block_for_this_branch = self.model_attn_search_cfg[label_of_num_of_attn_block][branch_id]
            num_of_attn_type_for_this_branch = self.model_attn_search_cfg[label_of_num_of_attn_type][branch_id]
            list_of_attn_type_for_this_branch = self.model_attn_search_cfg[label_of_list_of_attn_type][branch_id]

            list_of_num_of_attn_block_for_each_branch.append(num_of_attn_block_for_this_branch)
            list_of_num_of_attn_type_for_each_branch.append(num_of_attn_type_for_this_branch)

            self.inter_attn_search_infos[branch_id] = {label_of_num_of_attn_block: num_of_attn_block_for_this_branch}
            self.inter_attn_search_infos[branch_id].update({label_of_num_of_attn_type: num_of_attn_type_for_this_branch})
            self.inter_attn_search_infos[branch_id].update({label_of_list_of_attn_type: list_of_attn_type_for_this_branch})
            for attn_type_id, attn_type in zip(range(len(list_of_attn_type_for_this_branch)), list_of_attn_type_for_this_branch):
                if attn_type_id == 0:
                    self.inter_attn_search_infos[branch_id].update({label_of_attn_type_search_info: {attn_type: {label_of_weight_index: weight_idx}}})
                else:
                    self.inter_attn_search_infos[branch_id][label_of_attn_type_search_info].update({attn_type: {label_of_weight_index: weight_idx}})
                weight_idx += 1

        max_num_of_attn_block_for_any_branch = max(list_of_num_of_attn_block_for_each_branch)
        max_num_of_attn_type_for_any_branch = max(list_of_num_of_attn_type_for_each_branch)

        # weight parameter of inter-attention_types searching
        w_init = tf.random_normal_initializer()
        self.inter_attn_search_weights = tf.Variable(initial_value=1e-3 * w_init(shape=[num_of_branch, max_num_of_attn_block_for_any_branch, max_num_of_attn_type_for_any_branch], dtype='float32'),
                                                     trainable=True, name='inter_attn_search_weights')
        
        return [self.inter_attn_search_weights], self.inter_attn_search_infos
