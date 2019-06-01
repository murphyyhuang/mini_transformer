# coding=utf-8

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf


def _global_step(hparams):
  """Adjust global step if a multi-step optimizer is used."""
  step = tf.to_float(tf.train.get_or_create_global_step())
  multiplier = hparams.optimizer_multistep_accumulate_steps
  if not multiplier:
    return step

  tf.logging.info("Dividing global step by %d for multi-step optimizer."
                  % multiplier)
  return step / tf.to_float(multiplier)


def learning_rate_schedule(hparams):
  """Learning rate schedule based on hparams."""
  step_num = _global_step(hparams)
  schedule_string = hparams.learning_rate_schedule
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= learning_rate_factor(name, step_num, hparams)
  return ret


def learning_rate_factor(name, step_num, hparams):
  """Compute the designated learning rate factor from hparams."""
  if name == "constant":
    tf.logging.info("Base learning rate: %f", hparams.learning_rate_constant)
    return hparams.learning_rate_constant
  elif name == "linear_warmup":
    return tf.minimum(1.0, step_num / hparams.learning_rate_warmup_steps)
  elif name == "linear_decay":
    ret = (hparams.train_steps - step_num) / hparams.learning_rate_decay_steps
    return tf.minimum(1.0, tf.maximum(0.0, ret))
  elif name == "cosdecay":  # openai gpt
    in_warmup = tf.cast(step_num <= hparams.learning_rate_warmup_steps,
                        dtype=tf.float32)
    ret = 0.5 * (1 + tf.cos(
        np.pi * step_num / hparams.learning_rate_decay_steps))
    # if in warmup stage return 1 else return the decayed value
    return in_warmup * 1 + (1 - in_warmup) * ret
  elif name == "single_cycle_cos_decay":
    # Cosine decay to zero with a single cycle. This is different from
    # "cosdecay" because it starts at 1 when the warmup steps end.
    x = tf.maximum(step_num, hparams.learning_rate_warmup_steps)
    step = x - hparams.learning_rate_warmup_steps
    return tf.math.cos(
        step * np.pi / hparams.learning_rate_decay_steps) / 2.0 + 0.5
  elif name == "rsqrt_decay":
    return tf.rsqrt(tf.maximum(step_num, hparams.learning_rate_warmup_steps))
  elif name == "rsqrt_normalized_decay":
    scale = tf.sqrt(tf.to_float(hparams.learning_rate_warmup_steps))
    return scale * tf.rsqrt(tf.maximum(
        step_num, hparams.learning_rate_warmup_steps))
  elif name == "exp_decay":
    decay_steps = hparams.learning_rate_decay_steps
    warmup_steps = hparams.learning_rate_warmup_steps
    p = (step_num - warmup_steps) / decay_steps
    p = tf.maximum(p, 0.)
    if hparams.learning_rate_decay_staircase:
      p = tf.floor(p)
    return tf.pow(hparams.learning_rate_decay_rate, p)
  elif name == "rsqrt_hidden_size":
    return hparams.hidden_size ** -0.5
  # elif name == "legacy":
  #   return legacy_learning_rate_schedule(hparams)
  else:
    raise ValueError("unknown learning rate factor %s" % name)
