# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import tensorflow_probability as tfp

from pct.layers import common_layers


class MultiheadAttention(tf.keras.Model):

  def __init__(self,
               total_key_depth,
               total_value_depth,
               output_depth,
               num_heads=8,
               ):
    super(MultiheadAttention, self).__init__()

    self._total_key_depth = total_key_depth
    self._total_value_depth = total_value_depth
    self._output_depth = output_depth
    self._num_heads = num_heads

    self.q_dense_block = common_layers.Dense(self._total_key_depth, use_bias=False)
    self.k_dense_block = common_layers.Dense(self._total_key_depth, use_bias=False)
    self.v_dense_block = common_layers.Dense(self._total_value_depth, use_bias=False)

    self.multi_head_concat_block = common_layers.Dense(self._output_depth, use_bias=False)

  def call(self,
           query_antecedent,
           memory_antecedent,
           bias,
           training=False,
           mask=None,
           ):
    q = self.q_dense_block(query_antecedent, training)
    if memory_antecedent is None:
      memory_antecedent = query_antecedent
    k = self.k_dense_block(memory_antecedent, training)
    v = self.v_dense_block(memory_antecedent, training)

    q = split_heads(q, self._num_heads)
    k = split_heads(k, self._num_heads)
    v = split_heads(v, self._num_heads)

    key_depth_per_head = self._total_key_depth // self._num_heads
    q *= key_depth_per_head ** -0.5
    x = dot_product_attention(q, k, v, bias)
    x = combine_heads(x)

    x = self.multi_head_concat_block(x, training)

    return x

# [DEPRECATED tf.estimator]
# def multihead_attention(query_antecedent,
#                         memory_antecedent,
#                         total_key_depth,
#                         total_value_depth,
#                         output_depth,
#                         num_heads,
#                         cache=None,
#                         name='multihead_attention'):
#   with tf.variable_scope(name):
#     if cache is None or memory_antecedent is None:
#       q, k, v = compute_qkv(query_antecedent, None, total_key_depth, total_value_depth)
#
#     q = split_heads(q, num_heads)
#     if cache is None:
#       k = split_heads(k, num_heads)
#       v = split_heads(v, num_heads)
#
#     key_depth_per_head = total_key_depth // num_heads
#     q *= key_depth_per_head ** -0.5
#
#     x = dot_product_attention(q, k, v, None)
#     x = combine_heads(x)
#
#     x = common_layers.dense(
#       x, output_depth, use_bias=False, name="output_transform")
#
#     return x
#
#
# def compute_qkv(query_antecedent,
#                 memory_antecedent,
#                 total_key_depth,
#                 total_value_depth):
#   """
#
#   :param query_antecedent:
#   :param memory_antecedent:
#   :param total_key_depth:
#   :param total_value_depth:
#   :return:
#     q, k, v : [batch, length, depth] tensors
#   """
#
#   if memory_antecedent is None:
#     memory_antecedent = query_antecedent
#
#   q = compute_attention_component(query_antecedent, total_key_depth, "q")
#   k = compute_attention_component(memory_antecedent, total_key_depth, "k")
#   v = compute_attention_component(memory_antecedent, total_value_depth, "v")
#
#   return q, k, v
#
#
# def compute_attention_component(antecedent, total_depth, name='None'):
#   dense_result = common_layers.dense(antecedent, total_depth, use_bias=False, name=name)
#   return dense_result


def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = common_layers.shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])


def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = common_layers.shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          activation_type=None,
                          weight_dtype=None):

  logits = tf.matmul(q, k, transpose_b=True)
  if bias is not None:
    bias = common_layers.cast_like(bias, logits)
    logits += bias
  # if logits are fp16, upcast before softmax
  logits = maybe_upcast(logits, activation_type, weight_dtype)
  weights = tf.nn.softmax(logits, name="attention_weights")
  weights = common_layers.cast_like(weights, q)
  return tf.matmul(weights, v)


def mixed_precision_is_enabled(
    activation_dtype=None, weight_dtype=None, hparams=None):
  assert not (hparams and (activation_dtype or weight_dtype)), (
      "Provide only hparams or activation_dtype and weight_dtype")
  if (hparams and hasattr(hparams, "activation_dtype") and
      hasattr(hparams, "weight_dtype")):
    activation_dtype = hparams.activation_dtype
    weight_dtype = hparams.weight_dtype
  return activation_dtype == tf.float16 and weight_dtype == tf.float32


def maybe_upcast(logits,
                 activation_dtype=None, weight_dtype=None, hparams=None):
  if mixed_precision_is_enabled(activation_dtype, weight_dtype, hparams):
    return tf.cast(logits, tf.float32)
  return logits


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal


# [DEPRECATED tf.estimator]
# def get_layer_timing_signal_learned_1d(channels, layer, num_layers):
#   """get n-dimensional embedding as the layer (vertical) timing signal.
#
#   Adds embeddings to represent the position of the layer in the tower.
#
#   Args:
#     channels: dimension of the timing signal
#     layer: layer num
#     num_layers: total number of layers
#
#   Returns:
#     a Tensor of timing signals [1, 1, channels].
#   """
#   shape = [num_layers, 1, 1, channels]
#   layer_embedding = (
#       tf.get_variable(
#           "layer_embedding",
#           shape,
#           initializer=tf.random_normal_initializer(0, channels**-0.5)) *
#       (channels**0.5))
#   return layer_embedding[layer, :, :, :]


class GetLayerTimingSignalLearned1D(tf.keras.Model):

  def __init__(self, channels, num_layers):
    super(GetLayerTimingSignalLearned1D, self).__init__()
    shape = [num_layers, 1, 1, channels]
    # self._layer_embedding = (
    #   tf.get_variable(
    #       "layer_embedding",
    #       shape,
    #       initializer=tf.random_normal_initializer(0, channels**-0.5)) *
    #   (channels**0.5))
    self._layer_embedding = tf.get_variable(
      "layer_embedding",
      shape,
      initializer=tf.random_normal_initializer(),
    )

  def call(self, layer, training=False, mask=None):
    return self._layer_embedding[layer, :, :, :]


def encoder_decoder_attention_loss(expected_attention_logits,
                                   actual_attentions,
                                   loss_type="kl_divergence",
                                   loss_multiplier=1.0):
  """Computes encdec attention loss between expected and actual attentions.

  Args:
    expected_attention_logits: Tensor storing the expected encoder-decoder
      attention logits with shape [batch_size, target_length, input_length].
    actual_attentions: Dictionary with actual attention logits for different
      attention types and hidden layers.
    loss_type: type of the loss function.
    loss_multiplier: multiplier for the attention loss.

  Returns:
    KL_divergence loss between the actual and expected attention logits.
  """

  def combine_attentions(attention_list):
    """Combine different layer attentions and then average over layers/heads."""
    # Stack all hidden layer attention tensors to get a tensor with shape
    # [num_hidden_layers, batch_size, num_heads, target_length, input_length].
    attentions = tf.stack(attention_list)
    # Reduce mean across all layers (axis=0) and all heads (axis=2) to get a
    # tensor with shape [batch_size, target_length, input_length].
    return tf.reduce_mean(attentions, [0, 2])

  def kl_divergence_loss(expected_logits, actual_logits):
    p = tfp.distributions.Categorical(logits=expected_logits)
    q = tfp.distributions.Categorical(logits=actual_logits)
    return tfp.distributions.kl_divergence(p, q)

  def mse_loss(expected_logits, actual_weights):
    expected_weights = tf.nn.softmax(expected_logits)
    return tf.losses.mean_squared_error(expected_weights, actual_weights)

  # For each hidden layer, we have attention-logit and attention-weight tensors
  # with shape [batch_size, num_heads, target_length, input_length].
  loss = 0.0
  if loss_type == "mse":
    actual_encdec_attention_weights = [
        t for layer_key, t in actual_attentions.items()
        if "encdec_attention" in layer_key and not layer_key.endswith("/logits")
    ]
    actual_attention_weights = combine_attentions(
        actual_encdec_attention_weights)
    loss = mse_loss(expected_attention_logits, actual_attention_weights)
  else:
    actual_encdec_attention_logits = [
        t for layer_key, t in actual_attentions.items()
        if "encdec_attention" in layer_key and layer_key.endswith("/logits")
    ]
    actual_attention_logits = combine_attentions(actual_encdec_attention_logits)
    loss = kl_divergence_loss(expected_attention_logits,
                              actual_attention_logits)
  return loss * loss_multiplier


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  return attention_bias_local(length, -1, 0)


def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = common_layers.ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)

