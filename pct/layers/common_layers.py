# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from tensorflow.python.ops import control_flow_util
from tensorflow.python.framework import function


class LayerPrepostprocess(tf.keras.Model):

  def __init__(self,
               process_type,
               hparams,):
    super(LayerPrepostprocess, self).__init__()
    
    if process_type == "pre":
      self._process_sequence = hparams.layer_preprocess_sequence
    elif process_type == "post":
      self._process_sequence = hparams.layer_postprocess_sequence
    else:
      raise ValueError("Unknown input process_type. It should be selected in ['pre', 'post']")

    if 'n' in self._process_sequence:
      if hparams.norm_type == "layer":
        self.norm_block = LayerNorm(hparams.hidden_size, hparams.norm_epsilon)
      elif hparams.norm_type == "group":
        self.norm_block = GroupNorm(hparams.hidden_size, hparams.norm_epsilon)

  def call(self, previous_value=None, x=None, training=False):
    if self._process_sequence == 'none':
      return x
    for c in self._process_sequence:
      if c == 'a':
        x += previous_value
      elif c == 'n':
        x = self.norm_block(x, training)
      else:
        tf.logging.error('Unknown type of layer pre-post processing command.')
        raise ValueError
    return x


class LayerNorm(tf.keras.Model):
  def __init__(self, vector_dim, norm_epsilon):
    """Create Variables for layer norm."""
    super(LayerNorm, self).__init__()
    self._epsilon = norm_epsilon
    self._scale = tf.get_variable(
      "layer_norm_scale", [vector_dim], initializer=tf.ones_initializer())
    self._bias = tf.get_variable(
      "layer_norm_bias", [vector_dim], initializer=tf.zeros_initializer())

  def call(self, x, training=False, mask=None):
    epsilon, scale, bias = [cast_like(t, x) for t in [self._epsilon, self._scale, self._bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

    output = norm_x * scale + bias

    return output


class GroupNorm(tf.keras.Model):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""

  def __init__(self, vector_dim, norm_epsilon, num_groups=8):
    super(GroupNorm, self).__init__()
    self._epsilon = norm_epsilon
    self._num_groups = num_groups
    self._vector_dim = vector_dim

    # Prepare variables.
    self._scale = tf.get_variable(
      "group_norm_scale", [vector_dim], initializer=tf.ones_initializer())
    self._bias = tf.get_variable(
      "group_norm_bias", [vector_dim], initializer=tf.zeros_initializer())

  def call(self, x, training=False, mask=None):
    x_shape = shape_list(x)
    filters = x_shape[-1]
    assert filters == self._vector_dim, "Disagreement found here."
    epsilon, scale, bias = [cast_like(t, x) for t in [self._epsilon, self._scale, self._bias]]
    x = tf.reshape(x,  x_shape[:-1] + [self._num_groups, filters // self._num_groups])
    mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return tf.reshape(norm_x, x_shape) * scale + bias


class DenseReluDense(tf.keras.Model):

  def __init__(self, filter_size, output_size):
    super(DenseReluDense, self).__init__()
    self.dense_input_unit = Dense(
      filter_size,
      use_bias=True,
      activation=tf.nn.relu,
    )

    self.dense_output_unit = Dense(
      output_size,
      use_bias=True,
      activation=None,
    )

  def call(self, inputs, training=False, mask=None):
    hidden_mat = self.dense_input_unit(inputs, training)
    output_mat = self.dense_output_unit(hidden_mat, training)

    return output_mat


class Dense(tf.keras.Model):
  """
  Identical to tf.keras.layers.Dense
  """
  def __init__(self, units, **kwargs):
    super(Dense, self).__init__()
    self.activations = tf.keras.layers.Dense(units, **kwargs)

  def call(self, x, training=False, mask=None):
    return self.activations(x)


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i, dim in enumerate(static):
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def cast_like(x, y):
  """Cast x to y's dtype, if necessary."""
  x = tf.convert_to_tensor(x)
  y = tf.convert_to_tensor(y)

  if x.dtype.base_dtype == y.dtype.base_dtype:
    return x

  cast_x = tf.cast(x, y.dtype)
  if cast_x.device != x.device:
    x_name = "(eager Tensor)"
    try:
      x_name = x.name
    except AttributeError:
      pass
    tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                       x.device, cast_x.device)
  return cast_x


def gather(params, indices, dtype=tf.float32):
  """Version of tf.gather that works faster on tpu."""
  if not is_xla_compiled():
    return tf.gather(params, indices)
  vocab_size = params.get_shape().as_list()[0]
  indices_flat = tf.reshape(indices, [-1])
  out = tf.matmul(tf.one_hot(indices_flat, vocab_size, dtype=dtype), params)
  out = reshape_like(out, tf.expand_dims(indices, -1))
  return out


def is_xla_compiled():
  """Whether we are building graph that will be compiled by XLA.

  This checks whether the code is executing within an XLA context.

  If True, model authors should ensure the graph they build is compilable by
  XLA. Specifically, they should ensure that all ops have XLA implementations
  and that all shapes are statically known.

  Returns:
    bool, whether the current graph will be compiled for XLA.
  """
  ctxt = tf.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
  return control_flow_util.GetContainingXLAContext(ctxt) is not None


def reshape_like(a, b):
  """Reshapes a to match the shape of b in all but the last dimension."""
  ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
  if not tf.executing_eagerly():
    ret.set_shape(b.get_shape().as_list()[:-1] + a.get_shape().as_list()[-1:])
  return ret


def weights_nonzero(labels):
  """Assign weight 1.0 to all labels except for padding (id=0)."""
  return to_float(tf.not_equal(labels, 0))


def to_float(x):
  """Cast x to float; created because tf.to_float is deprecated."""
  return tf.cast(x, tf.float32)


def pad_with_zeros(logits, labels):
  """Pad labels on the length dimension to match logits length."""
  with tf.name_scope("pad_with_zeros", values=[logits, labels]):
    logits, labels = pad_to_same_length(logits, labels)
    if len(labels.shape) == 3:  # 2-d labels.
      logits, labels = pad_to_same_length(logits, labels, axis=2)
    return logits, labels


def pad_to_same_length(x, y, final_length_divisible_by=1, axis=1):
  """Pad tensors x and y on axis 1 so that they have the same length."""
  if axis not in [1, 2]:
    raise ValueError("Only axis=1 and axis=2 supported for now.")
  with tf.name_scope("pad_to_same_length", values=[x, y]):
    x_length = shape_list(x)[axis]
    y_length = shape_list(y)[axis]
    if (isinstance(x_length, int) and isinstance(y_length, int) and
        x_length == y_length and final_length_divisible_by == 1):
      return x, y
    max_length = tf.maximum(x_length, y_length)
    if final_length_divisible_by > 1:
      # Find the nearest larger-or-equal integer divisible by given number.
      max_length += final_length_divisible_by - 1
      max_length //= final_length_divisible_by
      max_length *= final_length_divisible_by
    length_diff1 = max_length - x_length
    length_diff2 = max_length - y_length

    def padding_list(length_diff, arg):
      if axis == 1:
        return [[[0, 0], [0, length_diff]],
                tf.zeros([tf.rank(arg) - 2, 2], dtype=tf.int32)]
      return [[[0, 0], [0, 0], [0, length_diff]],
              tf.zeros([tf.rank(arg) - 3, 2], dtype=tf.int32)]

    paddings1 = tf.concat(padding_list(length_diff1, x), axis=0)
    paddings2 = tf.concat(padding_list(length_diff2, y), axis=0)
    res_x = tf.pad(x, paddings1)
    res_y = tf.pad(y, paddings2)
    # Static shapes are the same except for axis=1.
    x_shape = x.shape.as_list()
    x_shape[axis] = None
    res_x.set_shape(x_shape)
    y_shape = y.shape.as_list()
    y_shape[axis] = None
    res_y.set_shape(y_shape)
    return res_x, res_y


def smoothing_cross_entropy(logits,
                            labels,
                            vocab_size,
                            confidence,
                            gaussian=False):
  """Cross entropy with label smoothing to limit over-confidence.

  Args:
    logits: Tensor of shape [batch_size, ?, ?, ?, vocab_size].
    labels: Tensor of shape [batch_size, ?, ?, ?].
    vocab_size: Tensor representing the size of the vocabulary.
    confidence: Used to determine on and off values for label smoothing.
      If `gaussian` is true, `confidence` is the variance to the Gaussian
      distribution.
    gaussian: Uses a Gaussian distribution for label smoothing

  Returns:
    Tensor of shape [batch_size, ?, ?, ?].
  """
  with tf.name_scope("smoothing_cross_entropy", values=[logits, labels]):
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / to_float(vocab_size - 1)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
        confidence * tf.log(confidence) + to_float(vocab_size - 1) *
        low_confidence * tf.log(low_confidence + 1e-20))

    if gaussian and confidence > 0.0:
      labels = tf.cast(labels, tf.float32)

      normal_dist = tfp.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
          tf.cast(tf.range(vocab_size), tf.float32)[:, None, None, None, None])
      # Reordering soft_targets from [vocab_size, batch_size, ?, ?, ?] to match
      # logits: [batch_size, ?, ?, ?, vocab_size]
      soft_targets = tf.transpose(soft_targets, perm=[1, 2, 3, 4, 0])
    else:
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=soft_targets)
    return xentropy - normalizing


def should_generate_summaries():
  """Is this an appropriate context to generate summaries.

  Returns:
    a boolean
  """
  name_scope = tf.contrib.framework.get_name_scope()
  if name_scope and "while/" in name_scope:
    # Summaries don't work well within tf.while_loop()
    return False
  if tf.get_variable_scope().reuse:
    # Avoid generating separate summaries for different data shards
    return False
  return True


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.

  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.

  Args:
    x: A `Tensor`.

  Returns:
    The input `Tensor`.
  """
  return x


def flatten4d3d(x):
  """Flatten a 4d-tensor into a 3d-tensor by joining width and height."""
  xshape = shape_list(x)
  result = tf.reshape(x, [xshape[0], xshape[1] * xshape[2], xshape[3]])
  return result


def shift_right_3d(x, pad_value=None):
  """Shift the second dimension of x right by one."""
  if pad_value is None:
    shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
  else:
    shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
  return shifted_targets


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
  """Matrix band part of ones.

  Args:
    rows: int determining number of rows in output
    cols: int
    num_lower: int, maximum distance backward. Negative values indicate
      unlimited.
    num_upper: int, maximum distance forward. Negative values indicate
      unlimited.
    out_shape: shape to reshape output by.

  Returns:
    Tensor of size rows * cols reshaped into shape out_shape.
  """
  if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
    # Needed info is constant, so we construct in numpy
    if num_lower < 0:
      num_lower = rows - 1
    if num_upper < 0:
      num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
      band = band.reshape(out_shape)
    band = tf.constant(band, tf.float32)
  else:
    band = tf.matrix_band_part(
        tf.ones([rows, cols]), tf.cast(num_lower, tf.int64),
        tf.cast(num_upper, tf.int64))
    if out_shape:
      band = tf.reshape(band, out_shape)

  return band

