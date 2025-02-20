import tensorflow as tf


class DARTS(object):
    """Differentiable Architecture Search"""
    def __init__(self, cfg, SearchNNmodel, optimizer, arch_optimizer, criterion, unrolled):
        self.cfg = cfg
        self.snn = SearchNNmodel
        self.optimizer = optimizer
        self.arch_optimizer = arch_optimizer
        self.criterion = criterion
        self.unrolled = unrolled


    def step(self, arch_train_inputs, arch_train_labels):
        if self.unrolled: # Using 'unrolled' method (using one-step unrolled validation loss)
            self._update_step_unrolled()
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


    def _update_step_unrolled(self):
        pass