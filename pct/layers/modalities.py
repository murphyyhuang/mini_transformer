# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.layers import common_layers
from pct.utils import hparams_lib


class SymbolBottomSimple(tf.keras.Model):

  def __init__(self, hparams, vocab_size):
    super(SymbolBottomSimple, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    hidden_dim = self._hparams.hidden_size

    var_name = "embedding_weights"
    self._embedding_space = tf.get_variable(
      var_name, [vocab_size, hidden_dim],
      initializer=tf.random_normal_initializer(0.0, hidden_dim ** -0.5)
    )

  def call(self, x, training=False, mask=None):

      # Ensure the inputs are 3-D
      if len(x.get_shape()) == 4:
        x = tf.squeeze(x, axis=3)
      while len(x.get_shape()) < 3:
        x = tf.expand_dims(x, axis=-1)

      ret = common_layers.gather(self._embedding_space, x)
      if self._hparams.multiply_embedding_mode == "sqrt_depth":
        ret *= self._hparams.hidden_size ** 0.5
      ret *= tf.expand_dims(
        common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
      return ret

  @property
  def embedding_space(self):
    return self._embedding_space


def symbol_top(body_output, target_embedding_space):
  body_output_shape = common_layers.shape_list(body_output)
  body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
  logits = tf.matmul(body_output, target_embedding_space, transpose_b=True)
  return tf.reshape(logits,
                    body_output_shape[:-1] + [1, target_embedding_space.shape[0]])


def generic_loss(top_out,
                 targets,
                 model_hparams,
                 weights_fn=common_layers.weights_nonzero,
                 gaussian=False):
  logits = top_out
  labels = targets

  # padded_cross_entropy
  confidence = 1.0 - model_hparams.label_smoothing
  logits_shape = common_layers.shape_list(logits)
  vocab_size = logits_shape[-1]

  logits, labels = common_layers.pad_with_zeros(logits, labels)
  logits = tf.reshape(
      logits,
      common_layers.shape_list(labels) + [vocab_size],
      name="padded_cross_entropy_size_check")
  logits = tf.cast(logits, tf.float32)
  xent = common_layers.smoothing_cross_entropy(
    logits, labels, vocab_size, confidence, gaussian=gaussian)
  weights = weights_fn(labels)
  return tf.reduce_sum(xent * weights), tf.reduce_sum(weights)
