# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import control_flow_util
from tensorflow.python.framework import function


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         norm_type,
                         depth,
                         epsilon,
                         default_name=None):
  with tf.variable_scope(default_name):
    if sequence == 'none':
      return x
    for c in sequence:
      if c == 'a':
        x += previous_value
      elif c == 'n':
        x = apply_norm(x, norm_type, depth, epsilon)
      else:
        tf.logging.error('Unknown type of layer pre-post processing command.')
        raise ValueError


def layer_preprocess(layer_input, hparams):
  assert "a" not in hparams.layer_preprocess_sequence, (
    "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
    None,
    layer_input,
    sequence=hparams.layer_preprocess_sequence,
    norm_type=hparams.norm_type,
    depth=None,
    epsilon=hparams.norm_epsilon,
    default_name='layer_prepostprocess')


def layer_postprocess(layer_input, layer_output, hparams):
  return layer_prepostprocess(
    layer_input,
    layer_output,
    sequence=hparams.layer_preprocess_sequence,
    norm_type=hparams.norm_type,
    depth=None,
    epsilon=hparams.norm_epsilon,
    default_name='layer_postprocess'
  )


def apply_norm(x, norm_type, depth, epsilon, layer_collection=None):
  """Apply Normalization."""
  if layer_collection is not None:
    assert norm_type == "layer"
  if norm_type == "layer":
    return layer_norm(
        x, filters=depth, epsilon=epsilon, layer_collection=layer_collection)
  if norm_type == "group":
    return group_norm(x, filters=depth, epsilon=epsilon)
  if norm_type == "none":
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")


def layer_norm(x,
               filters=None,
               epsilon=1e-6,
               name=None,
               reuse=None,
               layer_collection=None):
  """Layer normalize the tensor x, averaging over the last dimension."""
  if filters is None:
    filters = shape_list(x)[-1]
  with tf.variable_scope(
      name, default_name="layer_norm", values=[x], reuse=reuse):
    scale, bias = layer_norm_vars(filters)
    return layer_norm_compute(x, epsilon, scale, bias,
                              layer_collection=layer_collection)


def group_norm(x, filters=None, num_groups=8, epsilon=1e-5):
  """Group normalization as in https://arxiv.org/abs/1803.08494."""
  x_shape = shape_list(x)
  if filters is None:
    filters = x_shape[-1]
  assert len(x_shape) == 4
  assert filters % num_groups == 0
  # Prepare variables.
  scale = tf.get_variable(
      "group_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "group_norm_bias", [filters], initializer=tf.zeros_initializer())
  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  # Reshape and compute group norm.
  x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
  # Calculate mean and variance on heights, width, channels (not groups).
  mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return tf.reshape(norm_x, x_shape) * scale + bias


def dense_relu_dense(inputs, filter_size, output_size):
  h = dense(
    inputs,
    filter_size,
    use_bias=True,
    activation=tf.nn.relu,
    name='conv1',
  )

  o = dense(
    h,
    output_size,
    activation=None,
    use_bias=True,
    name='conv2',
  )
  return o


def layer_norm_vars(filters):
  """Create Variables for layer norm."""
  scale = tf.get_variable(
      "layer_norm_scale", [filters], initializer=tf.ones_initializer())
  bias = tf.get_variable(
      "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
  return scale, bias


def layer_norm_compute(x, epsilon, scale, bias, layer_collection=None):
  """Layer norm raw computation."""

  # Save these before they get converted to tensors by the casting below
  params = (scale, bias)

  epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
  mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
  variance = tf.reduce_mean(
      tf.squared_difference(x, mean), axis=[-1], keepdims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)

  output = norm_x * scale + bias

  return output


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


def dense(x, units, **kwargs):
  """Identical to layers.dense."""
  activations = tf.layers.Dense(units, **kwargs)(x)
  return activations


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

