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


def last_k_avg_max_min_metric(metric_list, curr_num_val, k):
  """calculate last k average/max/mix metric of validation"""
  if curr_num_val < k:
    k = curr_num_val
  else:
    # in case of restoring from checkpoint 
    if len(metric_list) < k:
      k = len(metric_list)

  return np.mean(metric_list[-k:]), np.amax(metric_list[-k:]), np.amin(metric_list[-k:])


def evaluation_result_writer(summary_writer, eval_mode, eval_loss, eval_result, step, val_metrics=None, topN_val_on_metrics=None, lr=None):
  with summary_writer.as_default():
    tf.summary.scalar(f"loss/{eval_mode}/criterion_loss", eval_loss, step=step)
    tf.summary.scalar(f"accuracy/{eval_mode}/accuracy", np.mean(eval_result['Accuracy']), step=step)
    tf.summary.scalar(f"precision/{eval_mode}/macroPrecision", np.mean(eval_result['Precision']), step=step)
    tf.summary.scalar(f"sensitivity/{eval_mode}/macroSensitivity", np.mean(eval_result['Sensitivity']), step=step)
    tf.summary.scalar(f"f1/{eval_mode}/macroF1", np.mean(eval_result['F1']), step=step)
    if eval_mode == 'train':
      tf.summary.scalar(f"learning_rate/{eval_mode}/learning_rate", lr, step=step)
    if eval_mode == 'validation':
      for metric in val_metrics:
        top1_metric_of_val = topN_val_on_metrics[metric].top_N_val_list[0][metric]
        if metric == 'Loss':
          tf.summary.scalar(f"loss/{eval_mode}/top-1_criterion_loss", top1_metric_of_val, step=step)
        elif metric == 'Accuracy':
          tf.summary.scalar(f"accuracy/{eval_mode}/top-1_accuracy", top1_metric_of_val, step=step)


def arch_evaluation_result_writer(summary_writer, eval_mode, eval_loss, step, lr=None):
  with summary_writer.as_default():
    tf.summary.scalar(f"loss/{eval_mode}/criterion_loss", eval_loss, step=step)
    tf.summary.scalar(f"learning_rate/{eval_mode}/learning_rate", lr, step=step)


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
        logging.info('Portion of training data used for training set: {}'.format(cfg['train_portion']))
        logging.info('Shuffle training data before splitting to create the selected training set: {}'.format(cfg['train_shuffle']))
        logging.info('Split the selected training set as validation data: {}'.format(cfg['val_from_train']))
        logging.info('Shuffle the selected training set before splitting to create the validation set: {}'.format(cfg['val_from_train_shuffle']))
        logging.info('Validation portion from the selected training set: {}'.format(cfg['val_portion']))
        logging.info('Dimension of resized image: {}'.format(cfg['resized_image_dim']))
        logging.info('Rescale input value: {}'.format(cfg['rescale_input_value']))
        if cfg['rescale_input_value']:
          logging.info('Rescale input value - Range: {}'.format(cfg['rescale_input_value_range']))
        logging.info('Normalize input value: {}'.format(cfg['normalization']))
        if cfg['normalization']:
          logging.info('Normalize input value - Statistics: {}'.format(cfg['normalization_stat']))
        logging.info('Data augmentation for training data: {}'.format(cfg['train_augment']))
        if cfg['train_augment']:
          logging.info('Train augmentation - Random horizontal flpping: {}'.format(cfg['train_augment_hflip']))
          logging.info('Train augmentation - Random vertical flpping: {}'.format(cfg['train_augment_vflip']))
          logging.info('Train augmentation - Random rotation: {}'.format(cfg['train_augment_rotation']))
          if cfg['train_augment_rotation']:
              logging.info('Train augmentation - Random rotation factor: {}'.format(cfg['train_augment_rotation_factor']))
        logging.info('Random oversampling for training data: {}'.format(cfg['train_random_oversampling']))
        if cfg['train_random_oversampling']:
          logging.info('Train random oversampling - Type: {}'.format(cfg['train_random_oversampling_type']))
          logging.info('Train random oversampling - Weight: {}'.format(cfg['train_random_oversampling_weight']))
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(ds_name))
    logging.info('-' * 100)


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
    logging.info('*** MODEL ATTENTION ***')
    logging.info('Model attention structure: {}'.format(cfg['model_attention']))
    logging.info('Model attention config path: {}'.format(cfg['model_attention_config_path']))
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
    logging.info('Weight decay: {}'.format(cfg['weights_decay']))
    logging.info('Training unit: {}'.format(cfg['train_unit']))
    logging.info('Maximum number of training unit: {}'.format(cfg['max_train_unit']))
    logging.info('Validation unit: {}'.format(cfg['val_unit']))
    logging.info('Validation on every unit: {}'.format(cfg['val_on_every_unit']))
    logging.info('Validation metrics used for find top-N validation (each metric considered separately): {}'.format(cfg['val_metrics_for_find_top_N_val']))
    logging.info('top-N Validation recorded on each metric: {}'.format(cfg['val_top_N']))
    logging.info('Validation last k metrics: {}'.format(cfg['val_last_k_metrics']))
    logging.info('Save unit: {}'.format(cfg['save_unit']))
    logging.info('Save on every unit: {}'.format(cfg['save_on_every_unit']))
    logging.info('Number of seed used: {}'.format(cfg['seed_num']))
    logging.info('Start seed number: {}'.format(cfg['seed_start']))


def write_model_config_summary(model_structure, cfg):
    """summarize important configuration of model"""
    logging.info('-' * 100)
    logging.info('Summary of model configuration')
    logging.info('-' * 100)
    logging.info('Model Structure: {}'.format(model_structure))
    logging.info('*** Detail of Input into model ***')
    logging.info('Input type for model: {}'.format(cfg['input_types_for_model']))
    logging.info('Number of input channel of input (data) tensor for model: {}'.format(cfg['number_of_input_channel']))
    logging.info('*** Detail of model ***')
    logging.info('Number of branch: {}'.format(cfg['number_of_branch']))
    logging.info('Indices of input channel(s) for each bracnch(es): {}'.format(cfg['indices_of_input_channel_for_branches']))
    logging.info('Dimension of resized input for each branch(es): {}'.format(cfg['resized_dim_of_input_for_branches']))
    logging.info('Color space conversion of input for each branch(es): {}'.format(cfg['color_space_conversion_for_branches']))
    logging.info('DCT transformation of input for each branch(es): {}'.format(cfg['dct_transformation_for_branches']))
    logging.info('Normalization of input for each branch(es): {}'.format(cfg['input_normalization_for_branches']))
    logging.info('Neural network configurations of branch(es): {}'.format(cfg['NN_config_of_branch']))
    logging.info('Feature aggregration from branch(es) of model: {}'.format(cfg['feature_aggregration_from_branches']))
    logging.info('Optional dense layer(s) before classification dense layer of model: {}'.format(cfg['optional_dense_layers_of_model']))
    logging.info('-' * 100)
    

def write_model_attention_config_summary(model_attn, cfg):
    """summarize important configuration of model attention"""
    logging.info('-' * 100)
    logging.info('Summary of model attention configuration')
    logging.info('-' * 100)
    logging.info('Model Attention: {}'.format(model_attn))
    logging.info('Attention existence: {}'.format(cfg['attention_existence']))
    if cfg['attention_existence']:
      logging.info('Attention configuration for each bracnch(es): {}'.format(cfg['attention_config_of_branches']))
    logging.info('-' * 100)


def write_exec_search_config_summary(cfg):
    """summarize important configuration of search execution"""
    logging.info('-' * 100)
    logging.info('Summary of search execution configuration')
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
    logging.info('*** MODEL ATTENTION SEARCH ***')
    logging.info('Model attention search structure: {}'.format(cfg['model_attention_search']))
    logging.info('Model attention search config path: {}'.format(cfg['model_attention_search_config_path']))
    logging.info('-' * 100)
    logging.info('*** OPTIMIZER ***')
    logging.info('Optimizer: {}'.format(cfg['optimizer']))
    logging.info('Optimizer configuration: {}'.format(cfg['optimizer_config']))
    logging.info('-' * 100)
    logging.info('*** LEARNING RATE SCHEDULER ***')
    logging.info('Learning rate schedule: {}'.format(cfg['lr_schedule']))
    logging.info('Learning rate configuration: {}'.format(cfg['lr_config']))
    logging.info('-' * 100)
    logging.info('*** ARCH OPTIMIZER ***')
    logging.info('Optimizer of Architecture Search: {}'.format(cfg['arch_optimizer']))
    logging.info('Optimizer configuration of Architecture Search: {}'.format(cfg['arch_optimizer_config']))
    logging.info('-' * 100)
    logging.info('*** ARCH LEARNING RATE SCHEDULER ***')
    logging.info('Learning rate schedule of Architecture Search: {}'.format(cfg['arch_lr_schedule']))
    logging.info('Learning rate configuration of Architecture Search: {}'.format(cfg['arch_lr_config']))
    logging.info('-' * 100)
    logging.info('*** LOSS FUNCTION ***')
    logging.info('Loss function: {}'.format(cfg['loss_fn']))
    logging.info('Loss function configuration: {}'.format(cfg['loss_fn_config']))
    logging.info('-' * 100)
    logging.info('*** Training Setting ***')
    logging.info('Batch size: {}'.format(cfg['batch_size']))
    logging.info('Weight decay: {}'.format(cfg['weights_decay']))
    logging.info('Weight decay of Architecture Search: {}'.format(cfg['arch_weights_decay']))
    logging.info('Gradient clip by norm of Architecture Search: {}'.format(cfg['arch_grad_clip_norm']))
    logging.info('Training unit: {}'.format(cfg['train_unit']))
    logging.info('Maximum number of training unit: {}'.format(cfg['max_train_unit']))
    logging.info('Starting number of training unit for architecture search: {}'.format(cfg['start_arch_search_train_unit']))
    logging.info('Save unit: {}'.format(cfg['save_unit']))
    logging.info('Save on every unit: {}'.format(cfg['save_on_every_unit']))
    logging.info('Number of seed used: {}'.format(cfg['seed_num']))
    logging.info('Start seed number: {}'.format(cfg['seed_start']))


def write_model_attention_search_config_summary(model_attn_search, cfg):
    """summarize important configuration of model attention searching"""
    logging.info('-' * 100)
    logging.info('Summary of model attention searching configuration')
    logging.info('-' * 100)
    logging.info('Model Attention Search: {}'.format(model_attn_search))
    logging.info('Number of attention block for each bracnch(es): {}'.format(cfg['number_of_attention_block']))
    logging.info('Number of attention type used for searching on any attention block for each bracnch(es): {}'.format(cfg['number_of_attention_type']))
    logging.info('List of attention type used for searching on any attention block for each bracnch(es): {}'.format(cfg['list_of_attention_type']))
    logging.info('Search configuration of each attention type for branch(es): {}'.format(cfg['search_config_of_attention_type']))
    logging.info('-' * 100)


def write_info_and_weight_parameter_for_attention_searching(level, info, weight):
    """summarize information and weight parameter for attention searching"""
    logging.info('-'*100)
    logging.info('Summary of information and weight parameter for {}-attention_types searching'.format(level))
    logging.info('-'*100)
    logging.info('Information for {}-attention_types searching: {}'.format(level, info))
    logging.info('Weight parameter for {}-attention_types searching: {}'.format(level, weight))
    logging.info('-'*100)