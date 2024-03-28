from absl import app, flags, logging
from absl.flags import FLAGS
import sys
import os
import numpy as np
import tensorflow as tf

from modules.utils import (load_yaml, set_memory_growth, write_exec_config_summary, 
                           write_dataset_config_summary, write_model_config_summary,
                           ProgressBar)
from modules.dataset.dataset_main import (gen_dataset)
from modules.model.model_main import (NNmodel)
from modules.optimizers import (getOptimizer)
from modules.losses import (getLossFn)
from modules.metric import (accumulated_result, compute_metric)

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
    #model.summary(print_fn=lambda x: logging.info(x))
    model.summary()

    # optimizer
    optimizer = getOptimizer(cfg)

    # loss function
    criterion = getLossFn(cfg)

    # load checkpoint
    checkpoint_dir = './checkpoints/train/' + cfg['config_summary']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'), 
#                                     optimizer=optimizer,
                                     model=model)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        logging.info('[*] Load ckpt from {} at step {}.'.format(manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        logging.info("[*] Training from scratch.")

    # define training step function for model
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)
            loss = criterion(labels, logits)
            grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return logits, loss
    
    # define validation and test step function for model
    @tf.function
    def val_test_step(inputs):
        return model(inputs, training=False)

    # training loop
    remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch,
                           checkpoint.step.numpy() % steps_per_epoch)
    results = accumulated_result()
    class_names = ds_cfg['class_name']
    logging.info('Number of steps per epoch: {}'.format(steps_per_epoch))
    logging.info('Total/Remain of training steps: {} ({:.2f} epochs)'.format(remain_steps, remain_steps/steps_per_epoch))
    logging.info('Start/Resume TRAINING...')

    for inputs, input_paths, labels in train_ds.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        epochs = ((steps - 1) // steps_per_epoch) + 1

        logits, loss = train_step(inputs, labels)
        
        results.update(input_paths, labels, logits)
        prog_bar.update('epoch={:d}/{:.2f}, loss={:.4f}'.format(epochs, max_epoch, loss.numpy()))

        # check if end epoch
        if steps % steps_per_epoch == 0:
#           eval_report = compute_metric(results.accum_y_true, results.accum_y_logit, results.accum_y_prob, results.accum_y_pred, class_names)
            eval_report = compute_metric(results.accum_y_true, results.accum_y_prob, results.accum_y_pred, class_names)
            print(' Accuracy: ' + str("{:0.4f}".format(np.mean(eval_report['Accuracy']))) +
                  ' Precision: ' + str("{:0.4f}".format(np.mean(eval_report['Precision']))) +
                  ' Sensitivity: ' + str("{:0.4f}".format(np.mean(eval_report['Sensitivity']))) +
                  ' F1: ' + str("{:0.4f}".format(np.mean(eval_report['F1']))))
            results.reset()
    
    # validation
    results.reset()
    for val_inputs, val_input_paths, val_labels in val_ds:
        val_logits = val_test_step(val_inputs)
        results.update(val_input_paths, val_labels, val_logits)
    eval_report = compute_metric(results.accum_y_true, results.accum_y_prob, results.accum_y_pred, class_names)
    print('Validation !!!')
    print(' Accuracy: ' + str("{:0.4f}".format(np.mean(eval_report['Accuracy']))) +
          ' Precision: ' + str("{:0.4f}".format(np.mean(eval_report['Precision']))) +
          ' Sensitivity: ' + str("{:0.4f}".format(np.mean(eval_report['Sensitivity']))) +
          ' F1: ' + str("{:0.4f}".format(np.mean(eval_report['F1']))))

    # test
    results.reset()
    for test_inputs, test_input_paths, test_labels in test_ds:
        test_logits = val_test_step(test_inputs)
        results.update(test_input_paths, test_labels, test_logits)
    eval_report = compute_metric(results.accum_y_true, results.accum_y_prob, results.accum_y_pred, class_names)
    print('Test !!!')
    print(' Accuracy: ' + str("{:0.4f}".format(np.mean(eval_report['Accuracy']))) +
          ' Precision: ' + str("{:0.4f}".format(np.mean(eval_report['Precision']))) +
          ' Sensitivity: ' + str("{:0.4f}".format(np.mean(eval_report['Sensitivity']))) +
          ' F1: ' + str("{:0.4f}".format(np.mean(eval_report['F1']))))


if __name__ == '__main__':
    app.run(main)
