# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_variable_initializer(hparams):
  """Get variable initializer from hparams."""
  if not hparams.initializer:
    return None

  if not tf.executing_eagerly():
    tf.logging.info("Using variable initializer: %s", hparams.initializer)
  if hparams.initializer == "orthogonal":
    return tf.orthogonal_initializer(gain=hparams.initializer_gain)
  elif hparams.initializer == "uniform":
    max_val = 0.1 * hparams.initializer_gain
    return tf.random_uniform_initializer(-max_val, max_val)
  elif hparams.initializer == "normal_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="normal")
  elif hparams.initializer == "uniform_unit_scaling":
    return tf.variance_scaling_initializer(
        hparams.initializer_gain, mode="fan_avg", distribution="uniform")
  elif hparams.initializer == "xavier":
    return tf.initializers.glorot_uniform()
  else:
    raise ValueError("Unrecognized initializer: %s" % hparams.initializer)
