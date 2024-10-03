from absl import logging
import tensorflow as tf

from modules.model.structure import (OneBranch_ThreeInputChannelRGB)
from modules.utils import (write_info_and_weight_parameter_for_attention_searching)
from modules.labels import *
from modules.model.layer.CustomLayer import (SplitSoftmax)


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
        # the information will be used for being detail when build and call attention type module for INTER-attention_types searching  
        self.inter_attn_search_parameters, self.inter_attn_search_infos = self._initialize_inter_attn_search_parameter_and_info()
        write_info_and_weight_parameter_for_attention_searching(label_of_inter, self.inter_attn_search_infos, self.inter_attn_search_parameters)

        # generate weight parameter and information for INTRA-attention_types searching to find which configurations of each attention type is the optimized option for each block attached of each branch on the model structure,
        # the dimension of this weight parameter is (number_of_branch, MAX(number_of_attention_block[each_branch]), MAX(SUM(number_of_searching_options[each_attention_type])[each_branch])), note that the number of this weight parameter might be created redundant (excess) that some of them might be not used, the reason is to simply create tensor with same dimension for every branches. 
        # the information will be used for being detail when build and call attention type module for INTRA-attention_types searching
        self.intra_attn_search_parameters, self.intra_attn_search_infos = self._initialize_intra_attn_search_parameter_and_info()
        write_info_and_weight_parameter_for_attention_searching(label_of_intra, self.intra_attn_search_infos, self.intra_attn_search_parameters)

        # dictionary for information of INTER- and INTRA-attention_types searching
        attn_search_infos_dicts = {label_of_inter: self.inter_attn_search_infos, 
                                   label_of_intra: self.intra_attn_search_infos}

        # build model
        logging.info(f'Building {self.name}...')
        cfg_model_structure = self.cfg['model_structure']
        if cfg_model_structure == 'OneBranch_ThreeInputChannelRGB':
            self.model = OneBranch_ThreeInputChannelRGB.genModel(cfg=self.cfg, 
                                                                 ds_cfg=self.ds_cfg, 
                                                                 model_cfg=self.model_cfg, 
                                                                 model_attn_cfg=None, 
                                                                 attn_search=True, 
                                                                 attn_search_infos=attn_search_infos_dicts)
        else:
            raise ValueError('Unknown \'{}\' model'.format(cfg_model_structure))

        ########## TEST ##################
        self.get_probability_of_attention_search_options()
        ########## TEST ##################
    

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
        #                                       },
        #    'list_of_split_weights': list      
        #   },
        # 'maximum_number_of_attention_block_for_any_branch': integer,
        # 'maximum_number_of_attention_type_for_any_branch': integer        
        self.inter_attn_search_infos = dict()

        list_of_num_of_attn_block_for_each_branch = list()
        list_of_num_of_attn_type_for_each_branch = list()

        for branch_id in range(1, num_of_branch+1):
            # index of weight parameter assigned for each attention type 
            weight_idx = 0

            # list of splitting number of weight parameters into groups for assigning to group of attention types (and group of unused/excess weights [if any]) for searching in order to do softmax on their values for each group during training
            list_of_split_weights_infos_for_this_branch = list()

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
            
            list_of_split_weights_infos_for_this_branch.append(num_of_attn_type_for_this_branch)
            self.inter_attn_search_infos[branch_id].update({label_of_list_of_split_weights: list_of_split_weights_infos_for_this_branch})

        max_num_of_attn_block_for_any_branch = max(list_of_num_of_attn_block_for_each_branch)
        max_num_of_attn_type_for_any_branch = max(list_of_num_of_attn_type_for_each_branch)
        self.inter_attn_search_infos[label_of_max_num_of_attn_block_for_any_branch] = max_num_of_attn_block_for_any_branch
        self.inter_attn_search_infos[label_of_max_num_of_attn_type_for_any_branch] = max_num_of_attn_type_for_any_branch

        for branch_id in range(1, num_of_branch+1):
            remaining_num_of_excessing_weights = max_num_of_attn_type_for_any_branch - self.inter_attn_search_infos[branch_id][label_of_num_of_attn_type]
            if remaining_num_of_excessing_weights > 0:
                self.inter_attn_search_infos[branch_id][label_of_list_of_split_weights].append(remaining_num_of_excessing_weights)

        # weight parameter of inter-attention_types searching
        w_init = tf.random_normal_initializer()
        self.inter_attn_search_weights = tf.Variable(initial_value=1e-3 * w_init(shape=[num_of_branch, max_num_of_attn_block_for_any_branch, max_num_of_attn_type_for_any_branch], dtype='float32'),
                                                     trainable=True, name='inter_attn_search_weights')
        
        return [self.inter_attn_search_weights], self.inter_attn_search_infos


    def _initialize_intra_attn_search_parameter_and_info(self):
        num_of_branch = self.model_cfg['number_of_branch']

        # information of intra-attention_types searching (dict)
        # for each branch_id: (dict)
        #   {
        #    'number_of_attention_block': integer,
        #    'number_of_attention_type': integer,
        #    'list_of_attention_type': list,
        #    'attention_type_search_info': dict {
        #                                        'attention_type_name (string)': dict {
        #                                                                              'name_of_search_option': list,
        #                                                                              'number_of_search_option': dict {
        #                                                                                                               'search_option_name (string)': list, (for search option name #1)
        #                                                                                                               .................................... (for search option name #N)
        #                                                                                                              },
        #                                                                              'value_of_search_option': dict {
        #                                                                                                              'search_option_name (string)': list, (for search option name #1)
        #                                                                                                              .................................... (for search option name #N)
        #                                                                                                             },
        #                                                                              'start_weight_index_of_search_option' : dict {
        #                                                                                                                            'search_option_name (string)': integer, (for search option name #1)
        #                                                                                                                            ....................................... (for search option name #N)
        #                                                                                                                           },
        #                                                                              'total_number_of_all_search_option': integer,
        #                                                                             }                                               
        #                                        **************************************************** (for attention type #1) *************************************************************************,
        #                                        ......................................................................................................................................................
        #                                        **************************************************** (for attention type #N) *************************************************************************,
        #    'total_number_of_search_option_from_all_attention_type': integer,
        #    'list_of_split_weights': list                                                                                
        #                                       }
        #
        #   },
        # 'maximum_number_of_attention_block_for_any_branch': integer,
        # 'maximum_number_of_search_option_accumulated_from_all_attention_types_for_any_branch': integer        
        self.intra_attn_search_infos = dict()

        list_of_num_of_attn_block_for_each_branch = list()
        list_of_num_of_search_option_accumulated_from_all_attn_types_for_each_branch = list()

        for branch_id in range(1, num_of_branch+1):
            # total number of search option accumulated from all attention_type
            total_num_of_search_option_accumulated_from_all_attn_types_for_this_branch = 0

            # list of splitting number of weight parameters into groups for assigning to each group of search options, depending on each searh option's name and its values, (and group of unused/excess weights [if any]) for searching in order to do softmax on their search option values for each group during training
            list_of_split_weights_infos_for_this_branch = list()

            # index of weight parameter assigned for each of searching options of each attention type 
            weight_idx = 0

            num_of_attn_block_for_this_branch = self.model_attn_search_cfg[label_of_num_of_attn_block][branch_id]
            num_of_attn_type_for_this_branch = self.model_attn_search_cfg[label_of_num_of_attn_type][branch_id]
            list_of_attn_type_for_this_branch = self.model_attn_search_cfg[label_of_list_of_attn_type][branch_id]

            list_of_num_of_attn_block_for_each_branch.append(num_of_attn_block_for_this_branch)
            
            self.intra_attn_search_infos[branch_id] = {label_of_num_of_attn_block: num_of_attn_block_for_this_branch}
            self.intra_attn_search_infos[branch_id].update({label_of_num_of_attn_type: num_of_attn_type_for_this_branch})
            self.intra_attn_search_infos[branch_id].update({label_of_list_of_attn_type: list_of_attn_type_for_this_branch})
            
            for attn_type_id, attn_type in zip(range(len(list_of_attn_type_for_this_branch)), list_of_attn_type_for_this_branch):
                list_of_name_of_search_option_for_this_attn_type = list(self.model_attn_search_cfg[label_of_search_config_of_attn_type][branch_id][attn_type].keys())
                if attn_type_id == 0:
                    self.intra_attn_search_infos[branch_id].update({label_of_attn_type_search_info: {attn_type: {label_of_name_of_search_option: list_of_name_of_search_option_for_this_attn_type}}})
                else:
                    self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info].update({attn_type: {label_of_name_of_search_option: list_of_name_of_search_option_for_this_attn_type}})

                # total number of search option accumulated from all search option name for this attention_type
                total_num_of_search_option_accumulated_from_all_search_option_name_for_this_attn_type = 0
                for search_option_id, search_option_name in zip(range(len(list_of_name_of_search_option_for_this_attn_type)), list_of_name_of_search_option_for_this_attn_type):
                    list_of_value_for_this_search_option_name = self.model_attn_search_cfg[label_of_search_config_of_attn_type][branch_id][attn_type][search_option_name]
                    len_list_of_value_for_this_search_option_name = len(list_of_value_for_this_search_option_name)
                    
                    if search_option_id == 0:
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type].update({label_of_num_of_search_option: {search_option_name: len_list_of_value_for_this_search_option_name}})
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type].update({label_of_value_of_search_option: {search_option_name: list_of_value_for_this_search_option_name}})
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type].update({label_of_start_weight_idx_of_search_option: {search_option_name: weight_idx}})
                    else:
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type][label_of_num_of_search_option].update({search_option_name: len_list_of_value_for_this_search_option_name})
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type][label_of_value_of_search_option].update({search_option_name: list_of_value_for_this_search_option_name})
                        self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type][label_of_start_weight_idx_of_search_option].update({search_option_name: weight_idx})
                    
                    weight_idx += len_list_of_value_for_this_search_option_name
                    total_num_of_search_option_accumulated_from_all_search_option_name_for_this_attn_type += len_list_of_value_for_this_search_option_name
                    list_of_split_weights_infos_for_this_branch.append(len_list_of_value_for_this_search_option_name)

                self.intra_attn_search_infos[branch_id][label_of_attn_type_search_info][attn_type].update({label_of_total_num_of_all_search_option: total_num_of_search_option_accumulated_from_all_search_option_name_for_this_attn_type})
                
                total_num_of_search_option_accumulated_from_all_attn_types_for_this_branch += total_num_of_search_option_accumulated_from_all_search_option_name_for_this_attn_type

            self.intra_attn_search_infos[branch_id].update({label_of_total_num_of_search_option_from_all_attn_type: total_num_of_search_option_accumulated_from_all_attn_types_for_this_branch})
            self.intra_attn_search_infos[branch_id].update({label_of_list_of_split_weights: list_of_split_weights_infos_for_this_branch})
            list_of_num_of_search_option_accumulated_from_all_attn_types_for_each_branch.append(total_num_of_search_option_accumulated_from_all_attn_types_for_this_branch)

        max_num_of_attn_block_for_any_branch = max(list_of_num_of_attn_block_for_each_branch)
        max_num_of_search_option_accumulated_from_all_attn_types_for_any_branch = max(list_of_num_of_search_option_accumulated_from_all_attn_types_for_each_branch)
        self.intra_attn_search_infos[label_of_max_num_of_attn_block_for_any_branch] = max_num_of_attn_block_for_any_branch
        self.intra_attn_search_infos[label_of_max_num_of_search_option_accumulated_from_all_attn_types_for_any_branch] = max_num_of_search_option_accumulated_from_all_attn_types_for_any_branch

        for branch_id in range(1, num_of_branch+1):
            remaining_num_of_excessing_weights = max_num_of_search_option_accumulated_from_all_attn_types_for_any_branch - self.intra_attn_search_infos[branch_id][label_of_total_num_of_search_option_from_all_attn_type]
            if remaining_num_of_excessing_weights > 0:
                self.intra_attn_search_infos[branch_id][label_of_list_of_split_weights].append(remaining_num_of_excessing_weights)

        # weight parameter of intra-attention_types searching
        w_init = tf.random_normal_initializer()
        self.intra_attn_search_weights = tf.Variable(initial_value=1e-3 * w_init(shape=[num_of_branch, max_num_of_attn_block_for_any_branch, max_num_of_search_option_accumulated_from_all_attn_types_for_any_branch], dtype='float32'),
                                                     trainable=True, name='intra_attn_search_weights')

        return [self.intra_attn_search_weights], self.intra_attn_search_infos


    def get_probability_of_attention_search_options(self):
        """get probability of options of INTER- and INTRA-attention_types searching"""
        
        # calculate probability of options of both INTER- and INTRA-attention_types searching for each branch of the model
        num_of_branch = self.model_cfg['number_of_branch']

        inter_attn_search_options_prob = dict()
        intra_attn_search_options_prob = dict() 

        for branch_id in range(1, num_of_branch+1):
            # probability of options of INTER-attenttion_types searching for this branch
            inter_num_attn_block_this_branch = self.inter_attn_search_infos[branch_id][label_of_num_of_attn_block]
            inter_size_splits_this_branch = self.inter_attn_search_infos[branch_id][label_of_list_of_split_weights]
            inter_attn_search_options_prob_this_branch = SplitSoftmax(num_blocks=inter_num_attn_block_this_branch,
                                                                      num_ops=sum(inter_size_splits_this_branch),
                                                                      size_splits=inter_size_splits_this_branch)(self.inter_attn_search_weights[branch_id-1]) # indices of 1st dimension (its size = num_branch) of self.inter_attn_search_weights start from 0 to num_branch-1, so branch_id need to be subtracted with 1

            # probability of options of INTRA-attenttion_types searching for this branch
            intra_num_attn_block_this_branch = self.intra_attn_search_infos[branch_id][label_of_num_of_attn_block]
            intra_size_splits_this_branch = self.intra_attn_search_infos[branch_id][label_of_list_of_split_weights]
            intra_attn_search_options_prob_this_branch = SplitSoftmax(num_blocks=intra_num_attn_block_this_branch,
                                                                      num_ops=sum(intra_size_splits_this_branch),
                                                                      size_splits=intra_size_splits_this_branch)(self.intra_attn_search_weights[branch_id-1]) # indices of 1st dimension (its size = num_branch) of self.inter_attn_search_weights start from 0 to num_branch-1, so branch_id need to be subtracted with 1

            inter_attn_search_options_prob[branch_id] = inter_attn_search_options_prob_this_branch
            intra_attn_search_options_prob[branch_id] = intra_attn_search_options_prob_this_branch
        
        ############# TEST ############
        print('#'*100)
        print('inter_attn_search_options_prob: {}'.format(inter_attn_search_options_prob))
        print('#'*100)
        print('intra_attn_search_options_prob: {}'.format(intra_attn_search_options_prob))
        ############# TEST ############