from absl import logging

from modules.algo_search.algo_search_methods.darts import (DARTS)


def algoSearch(cfg, ds_cfg, model_cfg, model_attn_search_cfg, seed, snn, optimizer, arch_optimizer, criterion, name='AlgoSearch'):
    """Algorithm of architecture searching"""
    arch_algorithm = cfg['arch_algorithm']
    arch_algorithm_config = cfg['arch_algorithm_config']
    logging.info(f'Configuring {name}: {arch_algorithm} ...')
    
    if arch_algorithm == 'DARTS':
        unrolled = arch_algorithm_config['unrolled']
        algo_search = DARTS(cfg=cfg,
                            ds_cfg=ds_cfg, 
                            model_cfg=model_cfg, 
                            model_attn_search_cfg=model_attn_search_cfg, 
                            seed=seed,
                            snn=snn, 
                            optimizer=optimizer, 
                            arch_optimizer=arch_optimizer, 
                            criterion=criterion, 
                            unrolled=unrolled)
    else:
        raise ValueError('Unknown \'{}\' algorithm of architecture search'.format(arch_algorithm))
    
    return algo_search