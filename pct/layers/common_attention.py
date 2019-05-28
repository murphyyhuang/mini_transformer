# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from pct.layers import common_layers


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        cache=None,
                        name='multihead_attention'):
  with tf.variable_scope(name):
    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, None, total_key_depth, total_value_depth)

    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head ** -0.5

    x = dot_product_attention(q, k, v, None)
    x = combine_heads(x)

    x = common_layers.dense(
      x, output_depth, use_bias=False, name="output_transform")

    return x


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth):
  """

  :param query_antecedent:
  :param memory_antecedent:
  :param total_key_depth:
  :param total_value_depth:
  :return:
    q, k, v : [batch, length, depth] tensors
  """

  if memory_antecedent is None:
    memory_antecedent = query_antecedent

  q = compute_attention_component(query_antecedent, total_key_depth, "q")
  k = compute_attention_component(memory_antecedent, total_key_depth, "k")
  v = compute_attention_component(memory_antecedent, total_value_depth, "v")

  return q, k, v


def compute_attention_component(antecedent, total_depth, name='None'):
  return common_layers.dense(antecedent, total_depth, use_bias=False, name=name)


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
                          name=None,
                          activation_type=None,
                          weight_dtype=None):
  with tf.variable_scope(
    name, default_name="dot_product_attention"):
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


def get_layer_timing_signal_learned_1d(channels, layer, num_layers):
  """get n-dimensional embedding as the layer (vertical) timing signal.

  Adds embeddings to represent the position of the layer in the tower.

  Args:
    channels: dimension of the timing signal
    layer: layer num
    num_layers: total number of layers

  Returns:
    a Tensor of timing signals [1, 1, channels].
  """
  shape = [num_layers, 1, 1, channels]
  layer_embedding = (
      tf.get_variable(
          "layer_embedding",
          shape,
          initializer=tf.random_normal_initializer(0, channels**-0.5)) *
      (channels**0.5))
  return layer_embedding[layer, :, :, :]
