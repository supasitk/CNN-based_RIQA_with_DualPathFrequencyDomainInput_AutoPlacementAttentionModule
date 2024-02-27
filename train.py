from absl import app, flags, logging
from absl.flags import FLAGS
import sys
import os
import tensorflow as tf

from modules.utils import (load_yaml, set_memory_growth, write_exec_config_summary, 
                           write_dataset_config_summary, write_model_config_summary)
from modules.dataset.dataset_main import (gen_dataset)
from modules.model.model_main import (NNmodel)

flags.DEFINE_string('cfg_path', './configs/train/riqa.yaml', 'config file path')


def main(_):
    #load execution configuration
    cfg = load_yaml(FLAGS.cfg_path)

    # add path of directory containing this python file to PYTHONPATH list
    sys.path.append(os.path.dirname(__file__))

    # set log level of tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # set logging
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    # set up logger configuration for logging training and validation details
    log_dir = './logs/train/' + cfg['config_summary'] + '/run_log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file('training.log', log_dir)

    # summarize important configuration of execution to log file
    write_exec_config_summary(cfg)

    # set memory growth in tensorflow
    set_memory_growth()

    # Set seed
    # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
    # You can use this utility to make almost any Keras program fully deterministic.
    if cfg['seed'] != None:
        tf.keras.utils.set_random_seed(cfg['seed'])
        logging.info('Seed {} is used.'.format(cfg['seed']))
    else:
        logging.info('Seed is not set.')
    
    # Enable XLA
    tf.config.optimizer.set_jit(True)

    # summarize important configuration of dataset to log file
    ds_name = cfg['dataset_name']
    ds_cfg_path = cfg['dataset_config_path']
    ds_cfg = load_yaml(ds_cfg_path)
    write_dataset_config_summary(ds_name, ds_cfg)

    # summarize important configuration of model to log file
    model_structure = cfg['model_structure']
    model_cfg_path = cfg['model_config_path']
    model_cfg = load_yaml(model_cfg_path)
    write_model_config_summary(model_structure, model_cfg)

    # generate train/validation/test dataloader of the RIQA dataset
    train_ds, train_ds_len, val_ds, val_ds_len, test_ds, test_ds_len = gen_dataset(ds_name, ds_cfg, cfg)
    
    # create iterator of dataloader
    train_ds_iter = iter(train_ds)
    val_ds_iter = iter(val_ds)
    test_ds_iter = iter(test_ds)

    # calculate number of steps per epoch, total number of steps, maximum number of epochs for training
    train_batch_size = int(cfg['batch_size'])
    steps_per_epoch = train_ds_len // train_batch_size
    total_steps = int(cfg['max_train_unit']) if cfg['train_unit'] == 'step' else int(cfg['max_train_unit'] * steps_per_epoch)
    max_epoch = total_steps / steps_per_epoch

    # define network
    model = NNmodel(cfg, ds_cfg, model_cfg)
    # show detail of network
    logging.info('-' * 30)
    logging.info('Network structure: {}'.format(model_structure))
    model.summary(print_fn=lambda x: logging.info(x))

if __name__ == '__main__':
    app.run(main)
