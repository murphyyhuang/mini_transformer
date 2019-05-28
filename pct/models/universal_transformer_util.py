# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
from pct.layers import common_layers, common_attention


def universal_transformer_encoder(encoder_input,
                                  hparams,
                                  name="encoder"):
  x = encoder_input
  with tf.variable_scope(name):
    ffn_unit = functools.partial(
      transformer_encoder_ffn_unit,
      hparams=hparams,
    )

    attention_unit = functools.partial(
      transformer_encoder_attention_unit,
      hparams=hparams,
    )

    x, extra_output = universal_transformer_layer(
      x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


def universal_transformer_decoder(decoder_input,
                                  encoder_output,
                                  hparams,
                                  name="decoder"):
  x = decoder_input
  with tf.variable_scope(name):
    ffn_unit = functools.partial(
      transformer_decoder_ffn_unit,
      hparams=hparams,
    )

    attention_unit = functools.partial(
      transformer_decoder_attention_unit,
      hparams=hparams,
      encoder_output=encoder_output,
    )

    x, extra_output = universal_transformer_layer(
      x, hparams, ffn_unit, attention_unit)

    return common_layers.layer_preprocess(x, hparams), extra_output


def transformer_encoder_ffn_unit(x,
                                 hparams):
  with tf.variable_scope("ffn"):
    y = common_layers.dense_relu_dense(
      x,
      hparams.filter_size,
      hparams.hidden_size
    )

    x = common_layers.layer_postprocess(x, y, hparams)

    return x


def transformer_encoder_attention_unit(x,
                                       hparams):
  with tf.variable_scope("self_attention"):
    y = common_attention.multihead_attention(
      common_layers.layer_preprocess(x, hparams),
      None,
      hparams.hidden_size,
      hparams.hidden_size,
      hparams.hidden_size,
      hparams.num_heads,
    )
    x = common_layers.layer_postprocess(x, y, hparams)
  return x


def transformer_decoder_ffn_unit(x,
                                 hparams):
  with tf.variable_scope("ffn"):
    y = common_layers.dense_relu_dense(
      x,
      hparams.filter_size,
      hparams.hidden_size
    )

    x = common_layers.layer_postprocess(x, y, hparams)

    return x


def transformer_decoder_attention_unit(x,
                                       hparams,
                                       encoder_output):
  with tf.variable_scope("self_attention"):
    y = common_attention.multihead_attention(
      common_layers.layer_preprocess(x, hparams),
      None,
      hparams.hidden_size,
      hparams.hidden_size,
      hparams.hidden_size,
      hparams.num_heads,
    )
    x = common_layers.layer_postprocess(x, y, hparams)
  if encoder_output is not None:
    with tf.variable_scope("encdec_attention"):
      y = common_attention.multihead_attention(
        common_layers.layer_preprocess(x, hparams),
        encoder_output,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.hidden_size,
        hparams.num_heads,
      )
      x = common_layers.layer_postprocess(x, y, hparams)
  return x


def universal_transformer_layer(x,
                                hparams,
                                ffn_unit,
                                attention_unit):

  with tf.variable_scope("universal_transformer_%s" % hparams.recurrence_type):
    ut_function, initializer = get_ut_layer(x, hparams, ffn_unit, attention_unit)
    output, _, extra_output = tf.foldl(
      ut_function, tf.range(hparams.num_rec_steps),
      initializer=initializer
    )

    return output, extra_output


def get_ut_layer(x,
                 hparams,
                 ffn_unit,
                 attention_unit):

  if hparams.recurrence_type == "basic":
    ut_initializer = (x, x, x)  # (state, input, memory)
    ut_function = functools.partial(
      universal_transformer_basic,
      hparams=hparams,
      ffn_unit=ffn_unit,
      attention_unit=attention_unit
    )
  else:
    raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)

  return ut_function, ut_initializer


def universal_transformer_basic(layer_inputs,
                                step,
                                hparams,
                                ffn_unit,
                                attention_unit):
  state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0, name="unstack")
  new_state = step_preprocess(state, step, hparams)

  for i in range(hparams.num_inrecurrence_layers):
    with tf.variable_scope("rec_layer_%d" % i):
      new_state = ffn_unit(attention_unit(new_state))

  return new_state, inputs, memory


def step_preprocess(x, step, hparams):
  """Preprocess the input at the beginning of each step.

  Args:
    x: input tensor
    step: step
    hparams: model hyper-parameters

  Returns:
    preprocessed input.

  """

  if hparams.add_position_timing_signal:
    x = add_position_timing_signal(x, hparams)

  if hparams.add_step_timing_signal:
    x = add_step_timing_signal(x, step, hparams)

  return x


def add_position_timing_signal(x, hparams):
  """Add n-dimensional embedding as the position (horizontal) timing signal.

  Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters

  Returns:
    a Tensor with the same shape as x.

  """

  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = common_attention.get_timing_signal_1d(length, channels)

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + common_layers.cast_like(signal, x)

  else:
    ValueError("Unknown timing signal add or concat type: %s"
               % hparams.add_or_concat_timing_signal)

  return x_with_timing


def add_step_timing_signal(x, step, hparams):

  if hparams.recurrence_type == "act":
    num_steps = hparams.act_max_steos
  else:
    num_steps = hparams.num_rec_steps
  channels = common_layers.shape_list(x)[-1]

  if hparams.step_timing_signal_type == "learned":
    signal = common_attention.get_layer_timing_signal_learned_1d(
      channels, step, num_steps
    )

  if hparams.add_or_concat_timing_signal == "add":
    x_with_timing = x + common_layers.cast_like(signal, x)

  return x_with_timing
