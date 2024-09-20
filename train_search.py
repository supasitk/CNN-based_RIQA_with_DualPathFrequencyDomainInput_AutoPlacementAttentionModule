from absl import app, flags, logging
from absl.flags import FLAGS
import sys
import os
# #import time
# #import datetime
# #import numpy as np
import tensorflow as tf

from modules.utils import (load_yaml, set_memory_growth, write_exec_search_config_summary,
                           write_dataset_config_summary, write_model_config_summary,
                           write_model_attention_search_config_summary)
# from modules.utils import (  
# #                             
# #                            ProgressBar, 
# #                           evaluation_result_log, last_k_avg_max_min_accuracy,
# #                           evaluation_result_writer)
from modules.dataset.dataset_main import (gen_dataset)
from modules.model_search.model_search_main import (SearchNNmodel)
# #from modules.optimizers import (getOptimizer)
# #from modules.losses import (getLossFn)
# #from modules.metric import (accumulated_result, compute_metric)

flags.DEFINE_string('cfg_path', './configs/train_search/riqa_search.yaml', 'config file path')


def main(_):
    # load execution configuration
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
    log_dir = './logs/train_search/' + cfg['config_summary'] + '/run_log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file('training.log', log_dir)

    # summarize important configuration of search execution to log file
    write_exec_search_config_summary(cfg)

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

    # summarize important configuration of model attention search to log file
    model_attn_search = cfg['model_attention_search']
    model_attn_search_cfg_path = cfg['model_attention_search_config_path']
    model_attn_search_cfg = load_yaml(model_attn_search_cfg_path)
    write_model_attention_search_config_summary(model_attn_search, model_attn_search_cfg)

    # generate train/validation dataloader of the RIQA dataset 
    # (last 2 return items ('test_ds' and 'test_ds_len') ignored because they are not used for train_search)
    train_ds, train_ds_len, val_ds, val_ds_len, _, _ = gen_dataset(ds_name, ds_cfg, cfg)
    
#     # calculate number of steps per epoch, total number of steps, maximum number of epochs for training
#     train_batch_size = int(cfg['batch_size'])
#     steps_per_epoch = train_ds_len // train_batch_size
#     total_steps = int(cfg['max_train_unit']) if cfg['train_unit'] == 'step' else int(cfg['max_train_unit'] * steps_per_epoch)
#     max_epoch = total_steps / steps_per_epoch

    # define network
    snn = SearchNNmodel(cfg, ds_cfg, model_cfg, model_attn_search_cfg)
    # show detail of network
    # logging.info('-' * 30)
    # logging.info('Network structure: {}'.format(model_structure))
    # #model.summary(print_fn=lambda x: logging.info(x))
    # snn.model.summary()

#     # optimizer
#     optimizer = getOptimizer(cfg, total_steps, steps_per_epoch)

#     # loss function
#     criterion = getLossFn(cfg)

#     # load checkpoint
#     checkpoint_dir = './checkpoints/train/' + cfg['config_summary']
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'), 
# #                                     optimizer=optimizer,
#                                      model=model)
#     manager = tf.train.CheckpointManager(checkpoint=checkpoint,
#                                          directory=checkpoint_dir,
#                                          max_to_keep=3)
#     if manager.latest_checkpoint:
#         checkpoint.restore(manager.latest_checkpoint)
#         logging.info('[*] Load ckpt from {} at step {}.'.format(manager.latest_checkpoint, checkpoint.step.numpy()))
#     else:
#         logging.info("[*] Training from scratch.")

#     # define training step function for model
#     @tf.function
#     def train_step(inputs, labels):
#         with tf.GradientTape() as tape:
#             logits = model(inputs, training=True)
#             loss = criterion(labels, logits)
#             grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#         return logits, loss
    
#     # define validation and test step function for model
#     @tf.function
#     def val_test_step(inputs):
#         return model(inputs, training=False)

#     # training loop
#     remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
#     prog_bar = ProgressBar(steps_per_epoch,
#                            checkpoint.step.numpy() % steps_per_epoch)
#     train_accumulated_results = accumulated_result()
#     val_accumulated_results = accumulated_result()
#     val_on_every_step = int(cfg['val_on_every_unit']) if cfg['val_unit'] == 'step' else int(cfg['val_on_every_unit'] * steps_per_epoch)
#     val_acc_list = []
#     val_last_k_accuracy = int(cfg['val_last_k_accuracy'])
#     val_best_acc = 0.
#     val_best_acc_step = 0
#     test_accumulated_results = accumulated_result()
#     class_names = ds_cfg['class_name']
#     summary_writer = tf.summary.create_file_writer('./logs/train/' + cfg['config_summary'])
#     save_on_every_step = int(cfg['save_on_every_unit']) if cfg['save_unit'] == 'step' else int(cfg['save_on_every_unit'] * steps_per_epoch)

#     logging.info('Number of steps per epoch: {}'.format(steps_per_epoch))
#     logging.info('Total/Remain of training steps: {} ({:.2f} epochs)'.format(remain_steps, remain_steps/steps_per_epoch))
#     logging.info('Start/Resume TRAINING...')

#     start_time = time.time()

#     # train and validation
#     for inputs, input_paths, labels in train_ds.take(remain_steps):
#         checkpoint.step.assign_add(1)
#         steps = checkpoint.step.numpy()
#         epochs = ((steps - 1) // steps_per_epoch) + 1

#         logits, loss = train_step(inputs, labels)
        
#         train_accumulated_results.update(input_paths, labels, logits)
#         prog_bar.update('epoch={:d}/{:.2f}, loss={:.4f}, lr={:.2e}, elapsed_time={} h:m:s'.format(
#                         epochs, max_epoch, loss.numpy(), optimizer.learning_rate.numpy(), str(datetime.timedelta(seconds=time.time()-start_time))))

#         # check if end each epoch of train
#         if steps % steps_per_epoch == 0:
#             loss = criterion(np.array(train_accumulated_results.accum_y_true), np.array(train_accumulated_results.accum_y_logit))
#             train_eval_result = compute_metric(train_accumulated_results.accum_y_true, 
#                                                train_accumulated_results.accum_y_prob,
#                                                train_accumulated_results.accum_y_pred, 
#                                                class_names)
#             evaluation_result_log('Train', loss, train_eval_result, (steps / steps_per_epoch))
#             evaluation_result_writer(summary_writer, 'train', loss, train_eval_result, steps, lr=optimizer.learning_rate.numpy())
#             train_accumulated_results.reset()

#         # check if validation
#         if (steps % val_on_every_step) == 0 and steps > 1:
#             val_accumulated_results.reset()
#             for val_inputs, val_input_paths, val_labels in val_ds:
#                 val_logits = val_test_step(val_inputs)
#                 val_accumulated_results.update(val_input_paths, val_labels, val_logits)
#             val_loss = criterion(np.array(val_accumulated_results.accum_y_true), np.array(val_accumulated_results.accum_y_logit))
#             val_eval_result = compute_metric(val_accumulated_results.accum_y_true, 
#                                              val_accumulated_results.accum_y_prob, 
#                                              val_accumulated_results.accum_y_pred, 
#                                              class_names)
#             evaluation_result_log('Validation', val_loss, val_eval_result, (steps / steps_per_epoch))
#             current_val_acc = np.mean(val_eval_result['Accuracy'])
            
#             # check if best accuracy of validation
#             is_current_val_best_acc = False 
#             if current_val_acc > val_best_acc:
#                 is_current_val_best_acc = True
#                 # delete last best weight of validation
#                 last_best_val_acc_weight_filepath = f"{checkpoint_dir}/bestEP{(val_best_acc_step / steps_per_epoch)}_onlyWeights.weights.h5"
#                 last_best_val_acc_model_filepath = f"{checkpoint_dir}/bestEP{(val_best_acc_step / steps_per_epoch)}_ModelWithWeights.keras"
#                 if os.path.exists(last_best_val_acc_weight_filepath):
#                     os.remove(last_best_val_acc_weight_filepath)
#                 if os.path.exists(last_best_val_acc_model_filepath):
#                     os.remove(last_best_val_acc_model_filepath)
#                 val_best_acc = current_val_acc
#                 val_best_acc_step = steps
#                 model.save_weights(f"{checkpoint_dir}/bestEP{(val_best_acc_step / steps_per_epoch)}_onlyWeights.weights.h5")
#                 model.save(f"{checkpoint_dir}/bestEP{(val_best_acc_step / steps_per_epoch)}_ModelWithWeights.keras")

#             evaluation_result_writer(summary_writer, 'validation', val_loss, val_eval_result, steps, is_best_val_acc=is_current_val_best_acc)

#             val_acc_list.append(current_val_acc)
#             val_last_k_avg_acc, val_last_k_max_acc, val_last_k_min_acc = last_k_avg_max_min_accuracy(val_acc_list, 
#                                                                                                      int(steps/val_on_every_step), 
#                                                                                                      val_last_k_accuracy)
#             logging.info(str("[Validation - epoch {}]".format((steps / steps_per_epoch))) +
#                          str(" New best validation accuracy: {}".format(is_current_val_best_acc)) +
#                          str(", Last {:d} avg|max|min validation accuracy: {:0.4f}|{:0.4f}|{:0.4f}".format(val_last_k_accuracy, val_last_k_avg_acc, val_last_k_max_acc, val_last_k_min_acc)) +
#                          str(", Best validation accuracy: {:0.4f} @ epoch {:0.2f} [step {:d}]".format(val_best_acc, (val_best_acc_step / steps_per_epoch), val_best_acc_step)))
        
#         # check if checkpoint
#         if steps % save_on_every_step == 0:
#             manager.save()
#             logging.info('[*] Save ckpt file at {}'.format(manager.latest_checkpoint))

#     manager.save()
#     logging.info('-' * 50)
#     logging.info('[*] End of training and Save ckpt file at {}'.format(manager.latest_checkpoint))
    
#     # test
#     # evaluate the best weight of validating model with test dataset
#     best_val_acc_weight_filepath = f"{checkpoint_dir}/bestEP{(val_best_acc_step / steps_per_epoch)}_onlyWeights.weights.h5"
#     logging.info('-' * 50)
#     logging.info('Evaluating the best weight [val_acc: {:0.4f} @ epoch/step: {:0.2f}/{:d} ] of validating model with test dataset...'.format(val_best_acc, (val_best_acc_step / steps_per_epoch), val_best_acc_step))
#     logging.info('Load the best weight @ {}'.format(best_val_acc_weight_filepath))
#     # load the best weight into model
#     model.load_weights(best_val_acc_weight_filepath)
#     test_accumulated_results.reset()
#     for test_inputs, test_input_paths, test_labels in test_ds:
#         test_logits = val_test_step(test_inputs)
#         test_accumulated_results.update(test_input_paths, test_labels, test_logits)
#     test_loss = criterion(np.array(test_accumulated_results.accum_y_true), np.array(test_accumulated_results.accum_y_logit))
#     test_eval_result = compute_metric(test_accumulated_results.accum_y_true, 
#                                       test_accumulated_results.accum_y_prob, 
#                                       test_accumulated_results.accum_y_pred, 
#                                       class_names)
#     evaluation_result_log('Test', test_loss, test_eval_result, (steps / steps_per_epoch))
#     evaluation_result_writer(summary_writer, 'test', test_loss, test_eval_result, steps)

#     # write execution summary
#     logging.info('-' * 100)
#     logging.info("[*] Total execution time = {} h:m:s".format(str(datetime.timedelta(seconds=time.time()-start_time))))
#     logging.info("[*] Execution done! save ckpt file at {}".format(manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)
