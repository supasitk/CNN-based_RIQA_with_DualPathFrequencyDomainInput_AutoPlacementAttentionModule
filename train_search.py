from absl import app, flags, logging
from absl.flags import FLAGS
import sys
import os
import time
import datetime
import numpy as np
import tensorflow as tf

from modules.utils import (load_yaml, set_memory_growth, write_exec_search_config_summary,
                           write_dataset_config_summary, write_model_config_summary,
                           write_model_attention_search_config_summary, ProgressBar,
                           evaluation_result_log, evaluation_result_writer, arch_evaluation_result_writer,
                           last_k_avg_max_min_metric)
from modules.dataset.dataset_main import (gen_dataset)
from modules.model_search.model_search_main import (SearchNNmodel)
from modules.optimizers import (getOptimizer)
from modules.losses import (LossFn)
from modules.metric import (accumulated_result, compute_metric, record_top_N_validation_for_specific_metric)
from modules.algo_search.algo_search_main import (algoSearch)

flags.DEFINE_string('cfg_path', './configs/train_search/riqa_search.yaml', 'config file path')
flags.DEFINE_integer('seed', 0, 'seed number')


def main(_):
    # seed number
    seed = FLAGS.seed

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
    log_dir = './logs/train_search/' + cfg['config_summary'] + f"/seed{seed}" + '/run_log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.get_absl_handler().use_absl_log_file('training_search.log', log_dir)

    # summarize important configuration of search execution to log file
    write_exec_search_config_summary(cfg)

    # set memory growth in tensorflow
    set_memory_growth()

    # Set seed
    # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
    # You can use this utility to make almost any Keras program fully deterministic.
    if seed != None:
        tf.keras.utils.set_random_seed(seed)
        logging.info('Seed {} is used.'.format(seed))
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
    train_ds, train_ds_len, arch_train_ds, arch_train_ds_len, _, _ = gen_dataset(ds_name, ds_cfg, model_cfg, cfg, train_search=True, seed=seed)
    
    # calculate number of steps per epoch, total number of steps, maximum number of epochs for training
    train_batch_size = int(cfg['batch_size'])
    # whether using random oversampling 
    # (non-oversampling uses 'number_of_train_data (total_num_img)' as 'one_epoch' while 
    #  oversampling uses 'number_of_majority-class_train_data' as 'one_epoch')
    using_oversampling = ds_cfg['train_random_oversampling']
    if using_oversampling:
        dict_key_list_num_img_of_each_class_id = 'list_num_img_of_each_class_id'
        prob_sampling_each_class_on_each_batch = ds_cfg['train_random_oversampling_weight']
        num_class = ds_cfg['class_num']

        idx_majority_class = np.argmax(train_ds_len[dict_key_list_num_img_of_each_class_id])
        if prob_sampling_each_class_on_each_batch == None:  # uniform distribution across classes
            prob_sampling_majority_class_on_each_batch = 1.0 / num_class
        else:
            prob_sampling_majority_class_on_each_batch = prob_sampling_each_class_on_each_batch[idx_majority_class]

        steps_per_epoch = int(np.ceil(train_ds_len[dict_key_list_num_img_of_each_class_id][idx_majority_class] / (train_batch_size * prob_sampling_majority_class_on_each_batch))) 
    else:
        steps_per_epoch = int(np.ceil(train_ds_len['total_num_img'] / train_batch_size))
    
    total_steps = int(cfg['max_train_unit']) if cfg['train_unit'] == 'step' else int(cfg['max_train_unit'] * steps_per_epoch)
    max_epoch = total_steps / steps_per_epoch

    # define network
    snn = SearchNNmodel(cfg, ds_cfg, model_cfg, model_attn_search_cfg, seed=seed)
    # show detail of network
    logging.info('-' * 30)
    logging.info('Network structure: {}'.format(model_structure))
    #model.summary(print_fn=lambda x: logging.info(x))
    snn.model.summary()

    # optimizer
    optimizer = getOptimizer(cfg, total_steps, steps_per_epoch, name='optimizer', arch_search=False)

    # optimizer of architecture search
    arch_optimizer = getOptimizer(cfg, total_steps, steps_per_epoch, name='arch_optimizer', arch_search=True)

    # loss function
    criterion = LossFn(cfg, ds_cfg, train_ds_len)

    # algorithm of architecture search
    arch_algorithm = cfg['arch_algorithm']
    arch_algo_search = algoSearch(cfg, ds_cfg, model_cfg, model_attn_search_cfg, seed, snn, optimizer, arch_optimizer, criterion)

    # load checkpoint
    checkpoint_dir = './checkpoints/train_search/' + cfg['config_summary'] + f"/seed{seed}/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'), 
#                                     optimizer=optimizer,
#                                     arch_optimizer=arch_optimizer,
                                     model=snn.model,
                                     alphas_inter=snn.inter_attn_search_weights,
                                     alphas_intra=snn.intra_attn_search_weights)
    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        logging.info('[*] Load ckpt from {} at step {}.'.format(manager.latest_checkpoint, checkpoint.step.numpy()))
    else:
        logging.info("[*] Training from scratch.")

    # calculate the step number for starting to train weights of model's architecture searching
    cfg_start_arch_search_train_unit = cfg['start_arch_search_train_unit']
    arch_search_start_steps = int(cfg_start_arch_search_train_unit) if cfg['train_unit'] == 'step' else int(cfg_start_arch_search_train_unit * steps_per_epoch)
    logging.info('[*] Searching model after steps: {} ({:.2f} epochs).'.format(arch_search_start_steps, arch_search_start_steps/steps_per_epoch))
    
    # training loop
    remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch,
                           checkpoint.step.numpy() % steps_per_epoch)
    train_accumulated_results = accumulated_result()
    arch_train_accumulated_results = accumulated_result()
    arch_metrics_for_find_top_N_arch_search = cfg['arch_metrics_for_find_top_N_arch_search']
    arch_top_N = cfg['arch_top_N']
    top_N_arch_on_metrics = dict()
    arch_metrics_list = dict()
    for metric in arch_metrics_for_find_top_N_arch_search:
        top_N_arch_on_metrics[metric] = record_top_N_validation_for_specific_metric(N=arch_top_N,
                                                                                    metric=metric)
        arch_metrics_list[metric] = list()
    arch_last_k_metrics = int(cfg['arch_last_k_metrics'])
    class_names = ds_cfg['class_name']
    summary_writer = tf.summary.create_file_writer('./logs/train_search/' + cfg['config_summary'] + f"/seed{seed}/")
    save_on_every_step = int(cfg['save_on_every_unit']) if cfg['save_unit'] == 'step' else int(cfg['save_on_every_unit'] * steps_per_epoch)
    iter_arch_train_ds = iter(arch_train_ds)

    logging.info('Number of steps per epoch: {}'.format(steps_per_epoch))
    logging.info('Total/Remain of training steps: {} ({:.2f} epochs)'.format(remain_steps, remain_steps/steps_per_epoch))
    logging.info('Start/Resume TRAINING...')

    start_time = time.time()

    # train and validation
    for inputs, input_paths, labels in train_ds.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()
        epochs = ((steps - 1) // steps_per_epoch) + 1

        # check if adjusting weights of model's architecture searching
        if steps > arch_search_start_steps:
            arch_train_inputs, arch_train_input_paths, arch_train_labels = next(iter_arch_train_ds)
            if arch_algorithm == 'DARTS':
                arch_logits, arch_loss = arch_algo_search.arch_step(inputs, labels, arch_train_inputs, arch_train_labels)

            arch_train_accumulated_results.update(arch_train_input_paths, arch_train_labels, arch_logits)

        logits, loss = arch_algo_search.train_step(inputs, labels)
        
        train_accumulated_results.update(input_paths, labels, logits)
        prog_bar.update('epoch={:d}/{:.2f}, loss={:.4f}, lr={:.2e}, elapsed_time={} h:m:s'.format(
                        epochs, max_epoch, loss.numpy(), optimizer.learning_rate.numpy(), str(datetime.timedelta(seconds=time.time()-start_time))))

        # check if end each epoch of train
        if steps % steps_per_epoch == 0:
            loss = criterion.compute_loss(np.array(train_accumulated_results.accum_y_true), np.array(train_accumulated_results.accum_y_logit), training=True)
            train_eval_result = compute_metric(train_accumulated_results.accum_y_true, 
                                               train_accumulated_results.accum_y_prob,
                                               train_accumulated_results.accum_y_pred, 
                                               class_names)
            evaluation_result_log('Train', loss, train_eval_result, (steps / steps_per_epoch))
            evaluation_result_writer(summary_writer, 'train', loss, train_eval_result, steps, lr=optimizer.learning_rate.numpy())
            train_accumulated_results.reset()

            # check if already starting arch_search
            if steps > arch_search_start_steps:
                arch_loss = criterion.compute_loss(np.array(arch_train_accumulated_results.accum_y_true), np.array(arch_train_accumulated_results.accum_y_logit), training=True)
                arch_train_eval_result = compute_metric(arch_train_accumulated_results.accum_y_true, 
                                                        arch_train_accumulated_results.accum_y_prob,
                                                        arch_train_accumulated_results.accum_y_pred, 
                                                        class_names)
                evaluation_result_log('Arch_Train', arch_loss, arch_train_eval_result, (steps / steps_per_epoch))

                # summarize information of arch_search result
                current_epoch = steps / steps_per_epoch
                current_arch_train_loss = arch_loss
                current_arch_train_acc = np.mean(arch_train_eval_result['Accuracy'])
                current_arch_train_macroPrecision = np.mean(arch_train_eval_result['Precision'])
                current_arch_train_macroSensitivity = np.mean(arch_train_eval_result['Sensitivity'])
                current_arch_train_macroF1 = np.mean(arch_train_eval_result['F1'])
                current_arch_train_qwk = np.mean(arch_train_eval_result['QWK'])

                current_arch_train_info = {'Epoch': current_epoch,
                                           'Loss': current_arch_train_loss,
                                           'Accuracy': current_arch_train_acc,
                                           'macroPrecision': current_arch_train_macroPrecision,
                                           'macroSensitivity': current_arch_train_macroSensitivity,
                                           'macroF1': current_arch_train_macroF1,
                                           'QWK': current_arch_train_qwk
                                          }

                # check if the current arch_search information is better than the previous top-N arch_search for each metric,
                # if yes (for each metric), updating new top-N arch_search of the metric.
                # also, showing average/max/min value for each metric from last K arch_search
                logging.info('[Arch_Train - epoch {}]: '.format((current_epoch)))
                for metric in arch_metrics_for_find_top_N_arch_search:
                    # check if updating new top-N arch_search of this metric
                    top_N_arch_on_metrics[metric].update_top_N_validation(current_arch_train_info)
                    logging.info('\tNew top-{} ArchSearch of {} found: {}'.format(arch_top_N, metric, top_N_arch_on_metrics[metric].is_current_val_metric_value_better))
                    top_N_arch_on_metrics[metric].show_top_N_validation()

                    # showing average/max/min value for each metric from last K arch_search
                    arch_metrics_list[metric].append(current_arch_train_info[metric])
                    arch_last_k_avg_metric, arch_last_k_max_metric, arch_last_k_min_metric = last_k_avg_max_min_metric(arch_metrics_list[metric],
                                                                                                                       int((steps-arch_search_start_steps)/steps_per_epoch),
                                                                                                                       arch_last_k_metrics)
                    logging.info('\tLast {:d} avg|max|min ArchSearch {}: {:0.4f}|{:0.4f}|{:0.4f}\n'.format(arch_last_k_metrics, metric, arch_last_k_avg_metric, arch_last_k_max_metric, arch_last_k_min_metric))

                arch_evaluation_result_writer(summary_writer, 'arch_train', arch_loss, arch_train_eval_result, steps, arch_metrics=arch_metrics_for_find_top_N_arch_search, topN_arch_on_metrics=top_N_arch_on_metrics, lr=arch_optimizer.learning_rate.numpy())
                arch_train_accumulated_results.reset()
                snn.write_probability_of_attention_search_options(filepath=log_dir + 'prob_attn_search_options.log', step=steps, steps_per_epoch=steps_per_epoch)
   
        # check if checkpoint
        if steps % save_on_every_step == 0:
            manager.save()
            logging.info('[*] Save ckpt file at {}'.format(manager.latest_checkpoint))

    manager.save()
    logging.info('-' * 50)
    logging.info('[*] End of training and Save ckpt file at {}'.format(manager.latest_checkpoint))
    
    # write execution summary
    logging.info('-' * 100)
    logging.info("[*] Total execution time = {} h:m:s".format(str(datetime.timedelta(seconds=time.time()-start_time))))
    logging.info("[*] Execution done! save ckpt file at {}".format(manager.latest_checkpoint))


if __name__ == '__main__':
    app.run(main)
