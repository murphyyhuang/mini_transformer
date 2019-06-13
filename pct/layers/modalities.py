# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.layers import common_attention
from pct.layers import common_layers
from pct.utils import hparams_lib


class SymbolBottomSimple(tf.keras.Model):

  def __init__(self, hparams, vocab_size):
    super(SymbolBottomSimple, self).__init__()

    self._hparams = hparams_lib.copy_hparams(hparams)
    hidden_dim = self._hparams.hidden_size
    # num_shards = self._hparams.symbol_modality_num_shards
    #
    # shards = []
    # for i in range(num_shards):
    #   shard_size = (vocab_size // num_shards) + (
    #     1 if i < vocab_size % num_shards else 0)
    #   var_name = "weights_%d" % i
    #   shards.append(
    #     tf.get_variable(
    #       var_name, [shard_size, hidden_dim],
    #       initializer=tf.random_normal_initializer(0.0, hidden_dim ** -0.5)))
    #
    # if num_shards == 1:
    #   self.ret = shards[0]
    # else:
    #   self.ret = tf.concat(shards, 0)

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


# [DEPRECATED tf.estimator]
# def symbol_bottom_simple(x, model_hparams, vocab_size, name):
#   """Bottom transformation for symbols."""
#   with tf.variable_scope(name):
#     # Ensure the inputs are 3-D
#     if len(x.get_shape()) == 4:
#       x = tf.squeeze(x, axis=3)
#     while len(x.get_shape()) < 3:
#       x = tf.expand_dims(x, axis=-1)
#
#     var = get_weights(model_hparams, vocab_size)
#     ret = common_layers.gather(var, x)
#     if model_hparams.multiply_embedding_mode == "sqrt_depth":
#       ret *= model_hparams.hidden_size**0.5
#     ret *= tf.expand_dims(
#         common_layers.cast_like(tf.not_equal(x, 0), ret), -1)
#     return ret


# [DEPRECATED tf.estimator]
# def get_weights(model_hparams, vocab_size, hidden_dim=None):
#   """Create or get concatenated embedding or softmax variable.
#
#   Args:
#     model_hparams: HParams, model hyperparmeters.
#     vocab_size: int, vocabulary size.
#     hidden_dim: dim of the variable. Defaults to _model_hparams' hidden_size
#
#   Returns:
#      a list of num_shards Tensors.
#   """
#   if hidden_dim is None:
#     hidden_dim = model_hparams.hidden_size
#   num_shards = model_hparams.symbol_modality_num_shards
#   shards = []
#   for i in range(num_shards):
#     shard_size = (vocab_size // num_shards) + (
#         1 if i < vocab_size % num_shards else 0)
#     var_name = "weights_%d" % i
#     shards.append(
#         tf.get_variable(
#             var_name, [shard_size, hidden_dim],
#             initializer=tf.random_normal_initializer(0.0, hidden_dim**-0.5)))
#   if num_shards == 1:
#     ret = shards[0]
#   else:
#     ret = tf.concat(shards, 0)
#   # Convert ret to tensor.
#   if not tf.executing_eagerly():
#     ret = common_layers.convert_gradient_to_tensor(ret)
#   return ret
#
#
# def symbol_top(body_output, model_hparams, vocab_size):
#   """Generate logits.
#
#   Args:
#     body_output: A Tensor with shape
#       [batch, p0, p1, model_hparams.hidden_size].
#     model_hparams: HParams, model hyperparmeters.
#     vocab_size: int, vocabulary size.
#
#   Returns:
#     logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
#   """
#
#   scope_name = "softmax"
#   reuse = False
#   with tf.variable_scope(scope_name, reuse=reuse):
#     body_output_shape = common_layers.shape_list(body_output)
#     var = get_weights(model_hparams, vocab_size, body_output_shape[-1])
#     body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
#     logits = tf.matmul(body_output, var, transpose_b=True)
#     return tf.reshape(logits,
#                       body_output_shape[:-1] + [1, vocab_size])


# class SymbolTop(tf.keras.Model):
#
#   def __init__(self, hparams, vocab_size, custom_variables):
#     super(SymbolTop, self).__init__()
#     self._hparams = hparams_lib.copy_hparams(hparams)
#     self._vocab_size = vocab_size
#     hidden_dim = self._hparams.hidden_size
#     num_shards = self._hparams.symbol_modality_num_shards
#
#     shards = []
#     for i in range(num_shards):
#       shard_size = (vocab_size // num_shards) + (
#         1 if i < vocab_size % num_shards else 0)
#       var_name = "weights_%d" % i
#       shards.append(
#         tf.get_variable(
#           var_name, [shard_size, hidden_dim],
#           initializer=tf.random_normal_initializer(0.0, hidden_dim ** -0.5)))
#
#     if num_shards == 1:
#       self.ret = shards[0]
#     else:
#       self.ret = tf.concat(shards, 0)
#
#     custom_variables.append(self.ret)
#
#   def call(self, body_output, training=False, mask=None):
#     body_output_shape = common_layers.shape_list(body_output)
#     body_output = tf.reshape(body_output, [-1, body_output_shape[-1]])
#     logits = tf.matmul(body_output, self.ret, transpose_b=True)
#     return tf.reshape(logits,
#                       body_output_shape[:-1] + [1, self._vocab_size])


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
  with tf.name_scope("padded_cross_entropy", values=[logits, labels]):
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
