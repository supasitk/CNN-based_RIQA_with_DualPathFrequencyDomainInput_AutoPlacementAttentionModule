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
                           evaluation_result_log, evaluation_result_writer, arch_evaluation_result_writer)
from modules.dataset.dataset_main import (gen_dataset)
from modules.model_search.model_search_main import (SearchNNmodel)
from modules.optimizers import (getOptimizer)
from modules.losses import (LossFn)
from modules.metric import (accumulated_result, compute_metric)
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
    arch_algo_search = algoSearch(cfg, snn, optimizer, arch_optimizer, criterion)

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

    # define training step function for weights of model's computational operations 
    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            logits = snn.model((inputs, *snn.inter_attn_search_parameters, *snn.intra_attn_search_parameters), training=True)
            loss = criterion.compute_loss(labels, logits, training=True)
        grads = tape.gradient(loss, snn.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, snn.model.trainable_variables))
        return logits, loss
    
    # define training step function for weights of model's architecture searching 
    # -- DARTS -- #
    # ---- [Using 'unrolled' method (using one-step unrolled validation loss)]
    # @tf.function
    # def darts_unrolled_arch_train_step(train_inputs, train_labels, arch_train_inputs, arch_train_labels, train_lr, train_optimizer):
    #     def _compute_unrolled_model(inputs, labels, lr, optimizer):
    #         # calculate loss
    #         logits = snn.model((inputs, *snn.inter_attn_search_parameters, *snn.intra_attn_search_parameters), training=True)
    #         loss = criterion.compute_loss(labels, logits, training=True)
    #         # get weight of trainable varaibles 
    #         weight_vars = get_variable_infos(snn.model.trainable_variables)[1]
    #         # get momentum (velocity) of trainable variables (only if using optimizer with momentum, such as SGD with momentum)
    #         try:
    #             momentum_vars = list()
    #             for m_var in optimizer.momentums: # for SGD optimizer, it has one variable `momentums` [list] (only set if `momentum` argument is not 0) for keeping momentum (velocity) of each trainable variables
    #                 momentum_vars.append(tf.math.multiply(m_var.numpy(), optimizer.momentum)) # for SGD optimizer, 'momemtum' attribute is value of momentum hyperparameter
    #         except:
    #             momentum_vars = list()
    #             for w_var in weight_vars:
    #                 momentum_vars.append(tf.zeros_like(w_var))

    #         assert len(weight_vars) == len(momentum_vars)


    #     unrolled_model = _compute_unrolled_model(train_inputs, train_labels, train_lr, train_optimizer)


    # training loop
    remain_steps = max(total_steps - checkpoint.step.numpy(), 0)
    prog_bar = ProgressBar(steps_per_epoch,
                           checkpoint.step.numpy() % steps_per_epoch)
    train_accumulated_results = accumulated_result()
    arch_train_accumulated_results = accumulated_result()
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
                arch_logits, arch_loss = arch_algo_search.step(inputs, labels, arch_train_inputs, arch_train_labels)

            arch_train_accumulated_results.update(arch_train_input_paths, arch_train_labels, arch_logits)

        logits, loss = train_step(inputs, labels)
        
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

            if steps > arch_search_start_steps:
                arch_loss = criterion.compute_loss(np.array(arch_train_accumulated_results.accum_y_true), np.array(arch_train_accumulated_results.accum_y_logit), training=True)
                arch_train_eval_result = compute_metric(arch_train_accumulated_results.accum_y_true, 
                                                        arch_train_accumulated_results.accum_y_prob,
                                                        arch_train_accumulated_results.accum_y_pred, 
                                                        class_names)
                evaluation_result_log('Arch_Train', arch_loss, arch_train_eval_result, (steps/ steps_per_epoch))
                arch_evaluation_result_writer(summary_writer, 'arch_train', arch_loss, arch_train_eval_result, steps, lr=arch_optimizer.learning_rate.numpy())
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
