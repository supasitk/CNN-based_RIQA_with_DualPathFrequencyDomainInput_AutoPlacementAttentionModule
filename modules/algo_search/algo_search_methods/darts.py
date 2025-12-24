import tensorflow as tf

from modules.model_search.model_search_main import (SearchNNmodel)


def _flatten_and_concat(xs):
    """Flatten multi-dimensional tensor of weights to 1-D tensor of weights and finally concatenate all of these to 1-D tensor"""
    return tf.concat([tf.reshape(x, shape=[-1]) for x in xs], axis=-1) 


class DARTS(object):
    """Differentiable Architecture Search"""
    def __init__(self, cfg, ds_cfg, model_cfg, model_attn_search_cfg, seed, snn, optimizer, arch_optimizer, criterion, unrolled):
        self.cfg = cfg
        self.ds_cfg = ds_cfg
        self.model_cfg = model_cfg
        self.model_attn_search_cfg = model_attn_search_cfg
        self.seed = seed
        self.snn = snn
        self.optimizer = optimizer
        self.arch_optimizer = arch_optimizer
        self.criterion = criterion
        self.unrolled = unrolled

        self.train_weight_decay = self.cfg['weights_decay']

        if self.unrolled:
            self.unrolled_snn = SearchNNmodel(self.cfg, self.ds_cfg, self.model_cfg, self.model_attn_search_cfg, seed=self.seed, name='Unrolled_SearchNNmodel')


    # define training step function for weights of model's computational operations 
    @tf.function
    def train_step(self, train_inputs, train_labels):
        with tf.GradientTape() as tape:
            logits = self.snn.model((train_inputs, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            loss = self.criterion.compute_loss(train_labels, logits, training=True)
        grads = tape.gradient(loss, self.snn.model.trainable_variables)
        if self.cfg['grad_clip_norm'] != None:
            grads = [(tf.clip_by_norm(grad, self.cfg['grad_clip_norm'])) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.snn.model.trainable_variables))
        return logits, loss


    # define training step function for weights of model's architecture searching 
    def arch_step(self, train_inputs, train_labels, arch_train_inputs, arch_train_labels):
        if self.unrolled: # Using 'unrolled' method (using one-step unrolled validation loss)
            logits, loss = self._update_arch_step_unrolled(train_inputs, train_labels, arch_train_inputs, arch_train_labels)
        else: # NOT using 'unrolled' method (not using one-step unrolled validation loss)
            logits, loss = self._update_arch_step(arch_train_inputs, arch_train_labels)
        return logits, loss

    @tf.function
    def _update_arch_step(self, arch_train_inputs, arch_train_labels):
        with tf.GradientTape() as tape:
            logits = self.snn.model((arch_train_inputs, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            loss = self.criterion.compute_loss(arch_train_labels, logits, training=True)
        attn_search_weights = self.snn.inter_attn_search_parameters + self.snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        grads = tape.gradient(loss, attn_search_weights)
        if self.cfg['arch_grad_clip_norm'] != None:
            grads = [(tf.clip_by_norm(grad, self.cfg['arch_grad_clip_norm'])) for grad in grads]
        self.arch_optimizer.apply_gradients(zip(grads, attn_search_weights))
        return logits, loss

    @tf.function
    def _update_arch_step_unrolled(self, train_inputs, train_labels, arch_train_inputs, arch_train_labels):
        # update all learnable parameters of self.unrolled_snn (weights of model's computational operations and weights of INTER/INTRA-attention_types searching)
        self._compute_unrolled_model(train_inputs, train_labels)
        
        # calculate unrolled_snn_loss via forward
        with tf.GradientTape(persistent=True) as tape:
            unrolled_logits = self.unrolled_snn.model((arch_train_inputs, *self.unrolled_snn.inter_attn_search_parameters, *self.unrolled_snn.intra_attn_search_parameters), training=True)
            unrolled_loss = self.criterion.compute_loss(arch_train_labels, unrolled_logits, training=True)
        # compute gradient(LOSSval(w', alpha), alpha), 1st part of Eq. 7
        # calculate gradients of weights of INTER/INTRA-attention_types searching for unrolled_snn
        unrolled_snn_attn_search_weights = self.unrolled_snn.inter_attn_search_parameters + self.unrolled_snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        unrolled_snn_arch_train_w_grads = tape.gradient(unrolled_loss, unrolled_snn_attn_search_weights)
        # if self.cfg['arch_grad_clip_norm'] != None:
        #    unrolled_snn_arch_train_w_grads = [(tf.clip_by_norm(grad, self.cfg['arch_grad_clip_norm'])) for grad in unrolled_snn_arch_train_w_grads]
        # compute 2nd part of Eq. 7 (or Eq. 8 being approximation of Eq. 7)
        # calculate gradients of weights of model's computational operations for unrolled_snn
        unrolled_snn_train_w_grads = tape.gradient(unrolled_loss, self.unrolled_snn.model.trainable_variables)
        implicit_grads = self._hessian_vector_product(unrolled_snn_train_w_grads, train_inputs, train_labels)

         # compute Eq. 7
        unrolled_snn_arch_train_w_grads_final = [tf.math.subtract(g, tf.math.multiply(ig, self.optimizer.learning_rate)) for g, ig in zip(unrolled_snn_arch_train_w_grads, implicit_grads)]
       
        # update weight parameters for INTER/INTRA-attention_types searching of snn_model
        snn_attn_search_weights = self.snn.inter_attn_search_parameters + self.snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        if self.cfg['arch_grad_clip_norm'] != None:
            unrolled_snn_arch_train_w_grads_final = [(tf.clip_by_norm(grad, self.cfg['arch_grad_clip_norm'])) for grad in unrolled_snn_arch_train_w_grads_final]
        self.arch_optimizer.apply_gradients(zip(unrolled_snn_arch_train_w_grads_final, snn_attn_search_weights))

        # release tape
        del tape

        return unrolled_logits, unrolled_loss

    @tf.function
    def _compute_unrolled_model(self, train_inputs, train_labels):
        # calculate train_loss via forward
        with tf.GradientTape() as tape:
            train_logits = self.snn.model((train_inputs, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            train_loss = self.criterion.compute_loss(train_labels, train_logits, training=True)
        
        # get current weights of model's trainable variables (weights of model's computational operations)
        # and flatten multi-dimensional tensor of weights to 1-D tensor of weights and finally concatenate all of these to 1-D tensor
        train_w_vars = _flatten_and_concat(self.snn.model.trainable_variables)

        # get momentum (velocity) value of model's trainable variables from train_optimizer (only if using optimizer with momentum, such as SGD with momentum)
        # and flatten multi-dimensional tensor weights' momentum value to 1-D tensor of weights' momentum value and finally concatenate all of these to 1-D tensor
        try:
            # for SGD optimizer, it has one variable `momentums` [list] (only set if `momentum` argument is not 0) for keeping momentum (velocity) of each trainable variables
            train_momentum_vars = _flatten_and_concat(self.optimizer.momentums)
            # for SGD optimizer, 'momemtum' attribute is value of momentum hyperparameter
            train_momentum_vars = tf.math.multiply(train_momentum_vars, self.optimizer.momentum)
        except:
            train_momentum_vars = tf.zeros_like(train_w_vars)
        
        # get derivatives (gradients) of weights of model's trainable variables (weights of model's computational operations)
        # and flatten multi-dimensional tensor weights' gradient value to 1-D tensor of weights' gradient value and finally concatenate all of these to 1-D tensor
        train_w_grads = _flatten_and_concat(tape.gradient(train_loss, self.snn.model.trainable_variables))
        # add derivatives (gradients) of weight_regularization_loss to the gradients of train_loss (from loss funciton, total_loss = train_loss + weight_regularization_loss)
        train_w_grads = tf.math.add(train_w_grads, tf.math.multiply(train_w_vars, self.train_weight_decay))

        # calculate new weights of model's trainable variables from updating by the gradients, the momentum value, and learning_rate
        # w = w - lr * (momentum + gradients)
        new_train_w_vars = tf.math.subtract(train_w_vars, tf.math.multiply(tf.math.add(train_momentum_vars, train_w_grads), self.optimizer.learning_rate))

        # update self.unrolled_snn with new_train_w_vars (new_theta)        
        self._update_unrolled_model_for_theta(new_train_w_vars)

    @tf.function
    def _update_unrolled_model_for_theta(self, theta):
        # copy weight parameter for INTER-attention_types searching from snn to unrolled_snn
        for unrolled_snn_inter_attn_search_weights, snn_inter_attn_search_weights in zip(self.unrolled_snn.inter_attn_search_parameters, self.snn.inter_attn_search_parameters):
            unrolled_snn_inter_attn_search_weights.assign(tf.identity(snn_inter_attn_search_weights))
        # copy weight parameter for INTRA-attention_types searching from snn to unrolled_snn
        for unrolled_snn_intra_attn_search_weights, snn_intra_attn_search_weights in zip(self.unrolled_snn.intra_attn_search_parameters, self.snn.intra_attn_search_parameters):
            unrolled_snn_intra_attn_search_weights.assign(tf.identity(snn_intra_attn_search_weights))

        # set weight of unrolled_snn' trainable_variables with the theta (new_train_w_vars) argument
        offset = 0
        for var in self.unrolled_snn.model.trainable_variables:
            var_length = tf.math.reduce_prod(var.shape)
            var.assign(tf.reshape(theta[offset:offset+var_length], shape=var.shape))
            offset += var_length

    @tf.function
    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """slightly touch vector value to estimate the gradient with respect to alpha refer to Eq. 8 for more details."""
        # calculate eplison (here is 'R') in Eq. 8 
        # flatten and concatenate vector, then calculate its 2-norm
        norm_of_vector = tf.norm(_flatten_and_concat(vector))
        R = tf.math.divide(r, norm_of_vector)

        # let 'w' be current weights of snn_model's trainable variables ('train_w_vars'), so train_w_vars = w
        # compute and assign w+ to train_w_vars, so train_w_vars = w+ = w + (R * grad),then 
        # compute gradient of alpha (attn_search_weights) (respect with w+ and train_inputs/labels) in Eq. 8 
        for var, grad in zip(self.snn.model.trainable_variables, vector):
            var.assign_add(tf.math.multiply(grad, R))
        with tf.GradientTape() as tape_p:
            logits_p = self.snn.model((input, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            loss_p = self.criterion.compute_loss(target, logits_p, training=True)
        snn_attn_search_weights_p = self.snn.inter_attn_search_parameters + self.snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        snn_alpha_grads_p = tape_p.gradient(loss_p, snn_attn_search_weights_p)

        # compute and assign w- to train_w_vars, so train_w_vars = w- = w+ - 2(R * grad) = w + (R * grad) - 2(R *grad) = w - (R * grad),then 
        # compute gradient of alpha (attn_search_weights) (respect with w- and train_inputs/labels) in Eq. 8 
        for var, grad in zip(self.snn.model.trainable_variables, vector):
            var.assign_sub(tf.math.multiply(grad, tf.math.multiply(R, 2.0)))
        with tf.GradientTape() as tape_n:
            logits_n = self.snn.model((input, *self.snn.inter_attn_search_parameters, *self.snn.intra_attn_search_parameters), training=True)
            loss_n = self.criterion.compute_loss(target, logits_n, training=True)
        snn_attn_search_weights_n = self.snn.inter_attn_search_parameters + self.snn.intra_attn_search_parameters # combination of lists (concatenate the two lists into one list)
        snn_alpha_grads_n = tape_n.gradient(loss_n, snn_attn_search_weights_n)

        # restore train_w_vars to w (train_w_vars = w = [w-] + (R * grad) = [w - (R * grad)] + (R * grad))
        for var, grad in zip(self.snn.model.trainable_variables, vector):
            var.assign_add(tf.math.multiply(grad, R))
        
        # compute Eq. 8
        return [tf.math.divide(tf.math.subtract(grad_p, grad_n), tf.math.multiply(R, 2.0)) for grad_p, grad_n in zip(snn_alpha_grads_p, snn_alpha_grads_n)]