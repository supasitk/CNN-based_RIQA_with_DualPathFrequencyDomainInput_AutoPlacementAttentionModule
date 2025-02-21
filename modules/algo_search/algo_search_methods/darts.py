import tensorflow as tf

from modules.utils import (get_variable_infos)


class DARTS(object):
    """Differentiable Architecture Search"""
    def __init__(self, cfg, SearchNNmodel, optimizer, arch_optimizer, criterion, unrolled):
        self.cfg = cfg
        self.snn = SearchNNmodel
        self.optimizer = optimizer
        self.arch_optimizer = arch_optimizer
        self.criterion = criterion
        self.unrolled = unrolled

        self.train_weight_decay = self.cfg['weights_decay']


    def step(self, train_inputs, train_labels, arch_train_inputs, arch_train_labels):
        if self.unrolled: # Using 'unrolled' method (using one-step unrolled validation loss)
            logits, loss = self._update_step_unrolled(train_inputs, train_labels, arch_train_inputs, arch_train_labels)
        else: # NOT using 'unrolled' method (not using one-step unrolled validation loss)
            logits, loss = self._update_step(arch_train_inputs, arch_train_labels)
        return logits, loss

    @tf.function
    def _update_step(self, arch_train_inputs, arch_train_labels):
        with tf.GradientTape() as tape:
            logits = self.snn.model((arch_train_inputs, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            loss = self.criterion.compute_loss(arch_train_labels, logits, training=True)
        attn_search_weights = self.snn.inter_attn_search_parameters + self.snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        grads = tape.gradient(loss, attn_search_weights)
        if self.cfg['arch_grad_clip_norm'] != None:
            grads = [(tf.clip_by_norm(grad, self.cfg['arch_grad_clip_norm'])) for grad in grads]
        self.arch_optimizer.apply_gradients(zip(grads, attn_search_weights))
        return logits, loss


    def _update_step_unrolled(self, train_inputs, train_labels, arch_train_inputs, arch_train_labels):
        train_w_vars_unrolled_model = self._compute_unrolled_model(train_inputs, train_labels)


    # @tf.function
    def _compute_unrolled_model(self, train_inputs, train_labels):
        # calculate train_loss
        with tf.GradientTape() as tape:
            train_logits = self.snn.model((train_inputs, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            train_loss = self.criterion.compute_loss(train_labels, train_logits, training=True)
        
        # get weights of model's trainable variables (weights of model's computational operations)
        train_w_vars = get_variable_infos(self.snn.model.trainable_variables)[1]

        # get momentum (velocity) value of model's trainable variables (only if using optimizer with momentum, such as SGD with momentum)
        try:
            train_momentum_vars = list()
            for m_var in self.optimizer.momentums: # for SGD optimizer, it has one variable `momentums` [list] (only set if `momentum` argument is not 0) for keeping momentum (velocity) of each trainable variables
                train_momentum_vars.append(tf.math.multiply(m_var.numpy(), self.optimizer.momentum)) # for SGD optimizer, 'momemtum' attribute is value of momentum hyperparameter
        except:
            train_momentum_vars = list()
            for w_var in train_w_vars:
                train_momentum_vars.append(tf.zeros_like(w_var))

        # get derivatives (gradients) of weights of model's trainable variables (weights of model's computational operations)
        train_grads = tape.gradient(train_loss, self.snn.model.trainable_variables)
        # add derivatives (gradients) of weight_regularization_loss to the gradients of train_loss (total_loss = train_loss + weight_regularization_loss)
        train_grads = tf.math.add(train_grads, tf.math.multiply(train_w_vars, self.train_weight_decay))

        # calculate new weights of model's trainable variables from updating with the gradients, the momentum value, and learning_rate
        # w = w - lr * (momentum + gradients)
        train_lr = self.optimizer.learning_rate.numpy()
        new_train_w_vars = tf.math.subtract(train_w_vars, tf.math.multiply(tf.math.add(train_momentum_vars, train_grads), train_lr))

        return new_train_w_vars