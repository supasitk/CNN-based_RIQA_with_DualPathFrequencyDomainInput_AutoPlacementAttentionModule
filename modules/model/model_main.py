from absl import logging

import tensorflow as tf


def NNmodel(cfg, ds_cfg, model_cfg, name='NNmodel'):
    """Neural Network Model"""
    logging.info(f'Building {name}...')