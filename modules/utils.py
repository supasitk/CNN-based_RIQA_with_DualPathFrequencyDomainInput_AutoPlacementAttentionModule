from absl import logging
import yaml
import os
import sys
import time
import csv
import numpy as np
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


def Check_Layer_Not_Loaded_with_PretrainedWeight(model, initial_weights):
  """check which layers of the model are NOT loaded (replaced) with the pretrained weights"""
  layer_list = []

  for layer, initial_weight in zip(model.layers, initial_weights):
    weight = layer.get_weights()
    if weight and all(tf.nest.map_structure(np.array_equal, weight, initial_weight)):
      layer_list.append(layer.name)

  logging.info('Layers of the model have NO pretrained weights: {}'.format(layer_list))


def evaluation_result_log(eval_mode, eval_loss, eval_result, epoch_num=None):
  """Logging evaluation result of execution"""
  logging.info(str("[{} - epoch {}]".format(eval_mode, epoch_num)) + 
               ' Loss: ' + str("{:0.4f}".format(eval_loss)) +
               ', Accuracy: ' + str("{:0.4f}".format(np.mean(eval_result['Accuracy']))) +
               ', macroPrecision: ' + str("{:0.4f}".format(np.mean(eval_result['Precision']))) +
               ', macroSensitivity: ' + str("{:0.4f}".format(np.mean(eval_result['Sensitivity']))) +
               ', macroF1: ' + str("{:0.4f}".format(np.mean(eval_result['F1']))))


def last_k_avg_max_min_accuracy(acc_list, curr_num_val, k):
  """calculate last k average accuracy"""
  if curr_num_val < k:
    k = curr_num_val
  else:
    # in case of restoring from checkpoint 
    if len(acc_list) < k:
      k = len(acc_list)

  return np.mean(acc_list[-k:]), np.amax(acc_list[-k:]), np.amin(acc_list[-k:])


def evaluation_result_writer(summary_writer, eval_mode, eval_loss, eval_result, step, is_best_val_acc=False, lr=None):
  with summary_writer.as_default():
    tf.summary.scalar(f"loss/{eval_mode}/criterion_loss", eval_loss, step=step)
    tf.summary.scalar(f"accuracy/{eval_mode}/accuracy", np.mean(eval_result['Accuracy']), step=step)
    tf.summary.scalar(f"precision/{eval_mode}/macroPrecision", np.mean(eval_result['Precision']), step=step)
    tf.summary.scalar(f"sensitivity/{eval_mode}/macroSensitivity", np.mean(eval_result['Sensitivity']), step=step)
    tf.summary.scalar(f"f1/{eval_mode}/macroF1", np.mean(eval_result['F1']), step=step)
    if eval_mode == 'validation' and is_best_val_acc:
       tf.summary.scalar(f"accuracy/{eval_mode}/best_accuracy", np.mean(eval_result['Accuracy']), step=step)
    if eval_mode == 'train':
       tf.summary.scalar(f"learning_rate/{eval_mode}/learning_rate", lr, step=step)


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
    logging.info('*** MODEL ***')
    logging.info('Model structure: {}'.format(cfg['model_structure']))
    logging.info('Model config path: {}'.format(cfg['model_config_path']))
    logging.info('-' * 100)
    logging.info('*** OPTIMIZER ***')
    logging.info('Optimizer: {}'.format(cfg['optimizer']))
    logging.info('Optimizer configuration: {}'.format(cfg['optimizer_config']))
    logging.info('-' * 100)
    logging.info('*** LEARNING RATE SCHEDULER ***')
    logging.info('Learning rate schedule: {}'.format(cfg['lr_schedule']))
    logging.info('Learning rate configuration: {}'.format(cfg['lr_config']))
    logging.info('-' * 100)
    logging.info('*** LOSS FUNCTION ***')
    logging.info('Loss function: {}'.format(cfg['loss_fn']))
    logging.info('Loss function configuration: {}'.format(cfg['loss_fn_config']))
    logging.info('-' * 100)
    logging.info('*** Training Setting ***')
    logging.info('Batch size: {}'.format(cfg['batch_size']))
    logging.info('Dropout rate: {}'.format(cfg['dropout_rate']))
    logging.info('Weight decay: {}'.format(cfg['weights_decay']))
    logging.info('Training unit: {}'.format(cfg['train_unit']))
    logging.info('Maximum number of training unit: {}'.format(cfg['max_train_unit']))
    logging.info('Validation unit: {}'.format(cfg['val_unit']))
    logging.info('Validation on every unit: {}'.format(cfg['val_on_every_unit']))
    logging.info('Validation last k accuracy: {}'.format(cfg['val_last_k_accuracy']))
    logging.info('Save unit: {}'.format(cfg['save_unit']))
    logging.info('Save on every unit: {}'.format(cfg['save_on_every_unit']))
    logging.info('Seed: {}'.format(cfg['seed']))


def write_model_config_summary(model_structure, cfg):
    """summarize important configuration of model"""
    logging.info('-' * 100)
    logging.info('Summary of model configuration')
    logging.info('-' * 100)
    logging.info('Model Structure: {}'.format(model_structure))
    
    # details of configuration sepearted for each model
    if model_structure == 'OneBranch_ThreeInputChannelRGB':
      logging.info('*** Input Channel ***')
      logging.info('Number of input channel of dataset: {}'.format(cfg['number_of_input_channel']))
      logging.info('Indices of input channel(s) for each bracnch(es): {}'.format(cfg['indices_of_input_channel_for_branches']))
      logging.info('*** Detail of branch ***')
      logging.info('Number of branch: {}'.format(cfg['number_of_branch']))
      logging.info('Name of branch(es): {}'.format(cfg['name_of_branches']))
      logging.info('Neural network for branch(es): {}'.format(cfg['model_of_branches']))
      logging.info('Number of layer for branch(es): {}'.format(cfg['numLayer_of_branches']))
      logging.info('Pretrained weight for branch(es): {}'.format(cfg['pretrainedWeight_of_branches']))
    else:
      raise ValueError('Unknown \'{}\' model'.format(model_structure))
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
        logging.info('Dimension of resized image: {}'.format(cfg['resized_image_dim']))
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(ds_name))
    logging.info('-' * 100)