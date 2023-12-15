from absl import logging
import yaml
import os
import csv
import tensorflow as tf


def load_yaml(load_path):
    """load yaml file"""
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    
    return loaded


def set_memory_growth():
    """set memory growth in tensorflow"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info("Detect {} Physical GPUs, {} Logical GPUs.".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)


def read_csv_to_list_of_dict(csv_file, delimiter):
    file = open(csv_file, "r")
    data = list(csv.DictReader(file, delimiter=delimiter))
    file.close()
    return data


def write_exec_config_summary(cfg):
    """summarize important configuration of execution"""
    logging.info('-' * 100)
    logging.info('Summary of execution configuration')
    logging.info('{}'.format(cfg['config_summary']))
    logging.info('-' * 100)
    logging.info('*** DATASET ***')
    logging.info('Dataset: {}'.format(cfg['dataset_name']))
    logging.info('Dataset config path: {}'.format(cfg['dataset_config_path']))
    logging.info('-' * 100)


def write_dataset_config_summary(ds_name, cfg):
    """summarize important configuration of dataset"""
    logging.info('-' * 100)
    logging.info('Summary of dataset configuration')
    logging.info('-' * 100)
    logging.info('Dataset name: {}'.format(ds_name))

    # details of configuration sepearted for each dataset
    if ds_name == 'EyeQ':
        logging.info('Root directory of dataset: {}'.format(cfg['dataset_root_dir']))
        logging.info('Directory of preprocessed dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'])))
        logging.info('Directory of preprocessed dataset for train: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'], cfg['dataset_preprocessed_train_dir_name'])))
        logging.info('Directory of preprocessed dataset for test: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'], cfg['dataset_preprocessed_test_dir_name'])))
        logging.info('Directory of original source code of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'])))
        logging.info('Directory of train/test image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_test_list_dir_name'])))
        logging.info('Pathname of train image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_test_list_dir_name'], cfg['dataset_src_train_list_filename'])))
        logging.info('Pathname of test image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_test_list_dir_name'], cfg['dataset_src_test_list_filename'])))

        
        
        
        
        
        logging.info('Validation portion from training set : {}'.format(cfg['val_portion']))
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(ds_name))

    logging.info('-' * 100)