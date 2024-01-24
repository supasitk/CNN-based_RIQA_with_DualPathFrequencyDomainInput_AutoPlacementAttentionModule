from absl import logging
import yaml
import os
import sys
import time
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


class ProgressBar(object):
  """A progress bar which can print the progress modified from
     https://github.com/hellock/cvbase/blob/master/cvbase/progress.py"""
  def __init__(self, task_num=0, completed=0, bar_width=15):
    self.task_num = task_num
    max_bar_width = self._get_max_bar_width()
    self.bar_width = (
      bar_width if bar_width <= max_bar_width else max_bar_width)
    self.completed = completed
    self.first_step = completed
    self.warm_up = False

  def _get_max_bar_width(self):
    if sys.version_info > (3, 3):
      from shutil import get_terminal_size
    else:
      from backports.shutil_get_terminal_size import get_terminal_size
    terminal_width, _ = get_terminal_size()
    max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
    if max_bar_width < 10:
      logging.info('terminal width is too small ({}), please consider '
                   'widen the terminal for better progressbar '
                   'visualization'.format(terminal_width))
      max_bar_width = 10
    return max_bar_width  
    
  def reset(self):
    """reset"""
    self.completed = 0
    self.fps = 0

  def update(self, inf_str=''):
    """update"""
    self.completed += 1

    if not self.warm_up:
      self.start_time = time.time() - 1e-1
      self.warm_up = True

    if self.completed > self.task_num:
      self.completed = self.completed % self.task_num
      self.start_time = time.time() - 1 / self.fps
      self.first_step = self.completed - 1
      sys.stdout.write('\n')

    elapsed = time.time() - self.start_time
    self.fps = (self.completed - self.first_step) / elapsed
    percentage = self.completed / float(self.task_num)
    mark_width = int(self.bar_width * percentage)
    bar_chars = '>' * mark_width + ' ' * (self.bar_width - mark_width)
    stdout_str = '\rtrain [{}] {}/{}, {}  {:.1f} step/sec'
    sys.stdout.write(stdout_str.format(
      bar_chars, self.completed, self.task_num, inf_str, self.fps))

    sys.stdout.flush()


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
        logging.info('*** Directories and files of EyePACS ***')
        logging.info('Root directory of EyePACS dataset: {}'.format(cfg['original_dataset_root_dir']))
        logging.info('Directory of EyePACS dataset for train: {}'.format(os.path.join(cfg['original_dataset_root_dir'], cfg['original_dataset_train_dir_name'])))
        logging.info('Directory of EyePACS dataset for validation: {}'.format(os.path.join(cfg['original_dataset_root_dir'], cfg['original_dataset_val_dir_name'])))
        logging.info('Directory of EyePACS dataset for test: {}'.format(os.path.join(cfg['original_dataset_root_dir'], cfg['original_dataset_test_dir_name'])))
        logging.info('*** Directories and files of EyeQ ***')
        logging.info('Root directory of EyeQ dataset: {}'.format(cfg['dataset_root_dir']))
        logging.info('Directory of preprocessed dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'])))
        logging.info('Directory of preprocessed dataset for train: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'], cfg['dataset_preprocessed_train_dir_name'])))
        logging.info('Directory of preprocessed dataset for validation: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'], cfg['dataset_preprocessed_val_dir_name'])))
        logging.info('Directory of preprocessed dataset for test: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_preprocessed_dir_name'], cfg['dataset_preprocessed_test_dir_name'])))
        logging.info('Directory of original source code of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'])))
        logging.info('Directory of train/val/test image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_val_test_list_dir_name'])))
        logging.info('Pathname of train image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_val_test_list_dir_name'], cfg['dataset_src_train_list_filename'])))
        logging.info('Pathname of validation image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_val_test_list_dir_name'], cfg['dataset_src_val_list_filename'])))
        logging.info('Pathname of test image list of dataset: {}'.format(os.path.join(cfg['dataset_root_dir'], cfg['dataset_src_dir_name'], cfg['dataset_src_train_val_test_list_dir_name'], cfg['dataset_src_test_list_filename'])))
        logging.info('Column name of csv file for filename of image: {}'.format(cfg['dataset_csv_info_column_name_of_image_filename']))
        logging.info('Column name of csv file for label of image: {}'.format(cfg['dataset_csv_info_column_name_of_image_label']))
        logging.info('*** Dataset information ***')
        logging.info('Number of classes: {}'.format(cfg['class_num']))
        logging.info('Name of classes: {}'.format(cfg['class_name']))
        logging.info('*** Data preprocessing ***')
        logging.info('Split training data as validation data: {}'.format(cfg['val_from_train']))
        logging.info('Shuffle training data before splitting into train and validation data: {}'.format(cfg['val_from_train_shuffle']))
        logging.info('Validation portion from training set: {}'.format(cfg['val_portion']))
        logging.info('Data augmentation for training data: {}'.format(cfg['train_augment']))
        if cfg['train_augment']:
            logging.info('Train augmentation - Random horizontal flpping: {}'.format(cfg['train_augment_hflip']))
            logging.info('Train augmentation - Random vertical flpping: {}'.format(cfg['train_augment_vflip']))
            logging.info('Train augmentation - Random rotation: {}'.format(cfg['train_augment_rotation']))
            if cfg['train_augment_rotation']:
                logging.info('Train augmentation - Random rotation factor: {}'.format(cfg['train_augment_rotation_factor']))
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(ds_name))

    logging.info('-' * 100)
