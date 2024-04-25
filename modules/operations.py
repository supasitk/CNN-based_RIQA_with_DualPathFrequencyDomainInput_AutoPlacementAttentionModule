import tensorflow as tf


def regularizer(weights_decay):
  """l2 regularizer"""
  return tf.keras.regularizers.l2(weights_decay)


def kernel_init(seed=None):
  """He normal initializer"""
  return tf.keras.initializers.he_normal(seed)