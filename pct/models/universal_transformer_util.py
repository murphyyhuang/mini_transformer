# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
from pct.layers import common_layers, common_attention


class UniversalTransformerEncoder(tf.keras.Model):

  def __init__(self, hparams):
    super(UniversalTransformerEncoder, self).__init__()
    # self._name_scope = default_name
    self._hparams = hparams
    # TODO: (Murphy) variable scope here may cause some bugs

    self.step_preprocess = StepPreprocess(self._hparams)

    self.attention_unit_preprocess = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )
    self.attention_unit = common_attention.MultiheadAttention(
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.num_heads,
    )
    self.attention_unit_postprocess = common_layers.LayerPrepostprocess(
      'post',
      self._hparams
    )

    self.ffn_unit_preprocess = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )
    self.ffn_unit = common_layers.DenseReluDense(
      self._hparams.filter_size,
      self._hparams.hidden_size
    )
    self.ffn_unit_postprocess = common_layers.LayerPrepostprocess(
      'post',
      self._hparams
    )

    self.encoder_normalizer = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )

  def call(self, x, training=False, mask=None):

    if self._hparams.recurrence_type == "basic":
      ut_initializer = (x, x, x)  # (state, input, memory)
      ut_encoder_unit = functools.partial(
        self._universal_transformer_basic,
        training=training,
      )
      output, _, extra_output = tf.foldl(
        ut_encoder_unit, tf.range(self._hparams.num_rec_steps),
        initializer=ut_initializer,
      )
      output = self.encoder_normalizer(None, output, training)
      return output, extra_output

  def _universal_transformer_basic(self, layer_inputs, step, training):
    state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0, name="unstack")
    new_state = self.step_preprocess(state, step, self._hparams)

    new_state_y = self.attention_unit_preprocess(None, new_state, training)
    new_state_y = self.attention_unit(new_state_y, None, training)
    new_state = self.attention_unit_postprocess(new_state, new_state_y, training)
    new_state_y = self.ffn_unit_preprocess(None, new_state, training)
    new_state_y = self.ffn_unit(new_state_y, training)
    new_state = self.ffn_unit_postprocess(new_state, new_state_y, training)

    return new_state, inputs, memory

  def _universal_transformer_act(self):
    pass


class UniversalTransformerDecoder(tf.keras.Model):

  def __init__(self, hparams):
    super(UniversalTransformerDecoder, self).__init__()
    self._hparams = hparams

    self.step_preprocess = StepPreprocess(self._hparams)

    self.attention_unit_preprocess = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )
    self.attention_unit = common_attention.MultiheadAttention(
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.num_heads,
    )
    self.attention_unit_postprocess = common_layers.LayerPrepostprocess(
      'post',
      self._hparams
    )

    self.ende_attention_unit_preprocess = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )
    self.ende_attention_unit = common_attention.MultiheadAttention(
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.hidden_size,
      self._hparams.num_heads,
    )
    self.ende_attention_unit_postprocess = common_layers.LayerPrepostprocess(
      'post',
      self._hparams
    )

    self.ffn_unit_preprocess = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )
    self.ffn_unit = common_layers.DenseReluDense(
      self._hparams.filter_size,
      self._hparams.hidden_size
    )
    self.ffn_unit_postprocess = common_layers.LayerPrepostprocess(
      'post',
      self._hparams
    )

    self.decoder_normalizer = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )

  def call(self, decoder_input, encoder_output, training=False, mask=None):

    if self._hparams.recurrence_type == "basic":
      x = decoder_input
      ut_initializer = (x, x, x)  # (state, input, memory)
      ut_decoder_unit = functools.partial(
        self._universal_transformer_basic,
        encoder_output=encoder_output,
        training=training,
      )
      output, _, extra_output = tf.foldl(
        ut_decoder_unit, tf.range(self._hparams.num_rec_steps),
        initializer=ut_initializer,
      )
      output = self.decoder_normalizer(None, output, training)
      return output, extra_output

  def _universal_transformer_basic(self, decoder_input, step, encoder_output, training):
    state, inputs, memory = tf.unstack(decoder_input, num=None, axis=0, name="unstack")
    new_state = self.step_preprocess(state, step)

    new_state_y = self.attention_unit_preprocess(None, new_state, training)
    new_state_y = self.attention_unit(new_state_y, None, training)
    new_state = self.attention_unit_postprocess(new_state, new_state_y, training)
    new_state_y = self.ende_attention_unit_preprocess(None, new_state, training)
    new_state_y = self.ende_attention_unit(new_state_y, encoder_output, training)
    new_state = self.ende_attention_unit_postprocess(new_state, new_state_y, training)
    new_state_y = self.ffn_unit_preprocess(None, new_state, training)
    new_state_y = self.ffn_unit(new_state_y, training)
    new_state = self.ffn_unit_postprocess(new_state, new_state_y, training)

    return new_state, inputs, memory

  def _universal_transformer_act(self):
    pass


# [DEPRECATED tf.estimator]
# def universal_transformer_encoder(encoder_input,
#                                   hparams,
#                                   name="encoder"):
#   x = encoder_input
#   with tf.variable_scope(name):
#     ffn_unit = functools.partial(
#       transformer_encoder_ffn_unit,
#       hparams=hparams,
#     )
#
#     attention_unit = functools.partial(
#       transformer_encoder_attention_unit,
#       hparams=hparams,
#     )
#
#     x, extra_output = universal_transformer_layer(
#       x, hparams, ffn_unit, attention_unit)
#
#     return common_layers.layer_preprocess(x, hparams), extra_output

# [DEPRECATED tf.estimator]
# def universal_transformer_decoder(decoder_input,
#                                   encoder_output,
#                                   hparams,
#                                   name="decoder"):
#   x = decoder_input
#   with tf.variable_scope(name):
#     ffn_unit = functools.partial(
#       transformer_decoder_ffn_unit,
#       hparams=hparams,
#     )
#
#     attention_unit = functools.partial(
#       transformer_decoder_attention_unit,
#       hparams=hparams,
#       encoder_output=encoder_output,
#     )
#
#     x, extra_output = universal_transformer_layer(
#       x, hparams, ffn_unit, attention_unit)
#
#     return common_layers.layer_preprocess(x, hparams), extra_output
#
#
# def transformer_encoder_ffn_unit(x,
#                                  hparams):
#   with tf.variable_scope("ffn"):
#     y = common_layers.dense_relu_dense(
#       x,
#       hparams.filter_size,
#       hparams.hidden_size
#     )
#
#     x = common_layers.layer_postprocess(x, y, hparams)
#
#     return x
#
#
# def transformer_encoder_attention_unit(x,
#                                        hparams):
#   with tf.variable_scope("self_attention"):
#     y = common_attention.multihead_attention(
#       common_layers.layer_preprocess(x, hparams),
#       None,
#       hparams.hidden_size,
#       hparams.hidden_size,
#       hparams.hidden_size,
#       hparams.num_heads,
#     )
#     x = common_layers.layer_postprocess(x, y, hparams)
#   return x
#
#
# def transformer_decoder_ffn_unit(x,
#                                  hparams):
#   with tf.variable_scope("ffn"):
#     y = common_layers.dense_relu_dense(
#       x,
#       hparams.filter_size,
#       hparams.hidden_size
#     )
#
#     x = common_layers.layer_postprocess(x, y, hparams)
#
#     return x
#
#
# def transformer_decoder_attention_unit(x,
#                                        hparams,
#                                        encoder_output):
#   with tf.variable_scope("self_attention"):
#     y = common_attention.multihead_attention(
#       common_layers.layer_preprocess(x, hparams),
#       None,
#       hparams.hidden_size,
#       hparams.hidden_size,
#       hparams.hidden_size,
#       hparams.num_heads,
#     )
#     x = common_layers.layer_postprocess(x, y, hparams)
#   if encoder_output is not None:
#     with tf.variable_scope("encdec_attention"):
#       y = common_attention.multihead_attention(
#         common_layers.layer_preprocess(x, hparams),
#         encoder_output,
#         hparams.hidden_size,
#         hparams.hidden_size,
#         hparams.hidden_size,
#         hparams.num_heads,
#       )
#       x = common_layers.layer_postprocess(x, y, hparams)
#   return x
#
#
# def universal_transformer_layer(x,
#                                 hparams,
#                                 ffn_unit,
#                                 attention_unit):
#
#   with tf.variable_scope("universal_transformer_%s" % hparams.recurrence_type):
#     ut_function, initializer = get_ut_layer(x, hparams, ffn_unit, attention_unit)
#     output, _, extra_output = tf.foldl(
#       ut_function, tf.range(hparams.num_rec_steps),
#       initializer=initializer
#     )
#
#     return output, extra_output
#
#
# def get_ut_layer(x,
#                  hparams,
#                  ffn_unit,
#                  attention_unit):
#
#   if hparams.recurrence_type == "basic":
#     ut_initializer = (x, x, x)  # (state, input, memory)
#     ut_function = functools.partial(
#       universal_transformer_basic,
#       hparams=hparams,
#       ffn_unit=ffn_unit,
#       attention_unit=attention_unit
#     )
#   else:
#     raise ValueError("Unknown recurrence type: %s" % hparams.recurrence_type)
#
#   return ut_function, ut_initializer
#
#
# def universal_transformer_basic(layer_inputs,
#                                 step,
#                                 hparams,
#                                 ffn_unit,
#                                 attention_unit):
#   state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0, name="unstack")
#   new_state = step_preprocess(state, step, hparams)
#
#   for i in range(hparams.num_inrecurrence_layers):
#     with tf.variable_scope("rec_layer_%d" % i):
#       new_state = ffn_unit(attention_unit(new_state))
#
#   return new_state, inputs, memory


# [DEPRECATED tf.estimator]
# def step_preprocess(x, step, hparams):
#   """Preprocess the input at the beginning of each step.
#
#   Args:
#     x: input tensor
#     step: step
#     hparams: model hyper-parameters
#
#   Returns:
#     preprocessed input.
#
#   """
#
#   if hparams.add_position_timing_signal:
#     x = add_position_timing_signal(x, hparams)
#
#   if hparams.add_step_timing_signal:
#     x = add_step_timing_signal(x, step, hparams)
#
#   return x
#
#
# def add_position_timing_signal(x, hparams):
#   """Add n-dimensional embedding as the position (horizontal) timing signal.
#
#   Args:
#     x: a tensor with shape [batch, length, depth]
#     step: step
#     hparams: model hyper parameters
#
#   Returns:
#     a Tensor with the same shape as x.
#
#   """
#
#   length = common_layers.shape_list(x)[1]
#   channels = common_layers.shape_list(x)[2]
#   signal = common_attention.get_timing_signal_1d(length, channels)
#
#   if hparams.add_or_concat_timing_signal == "add":
#     x_with_timing = x + common_layers.cast_like(signal, x)
#     return x_with_timing
#
#   else:
#     ValueError("Unknown timing signal add or concat type: %s"
#                % hparams.add_or_concat_timing_signal)
#
#
# def add_step_timing_signal(x, step, hparams):
#
#   if hparams.recurrence_type == "act":
#     num_steps = hparams.act_max_steps
#   else:
#     num_steps = hparams.num_rec_steps
#   channels = common_layers.shape_list(x)[-1]
#
#   if hparams.step_timing_signal_type == "learned":
#     signal = common_attention.get_layer_timing_signal_learned_1d(
#       channels, step, num_steps
#     )
#
#   if hparams.add_or_concat_timing_signal == "add":
#     x_with_timing = x + common_layers.cast_like(signal, x)
#
#   return x_with_timing


class StepPreprocess(tf.keras.Model):
  """Preprocess the input at the beginning of each step.

  Args:
    hparams: model hyper-parameters

  Returns:
    preprocessed input.

  """
  def __init__(self, hparams):
    super(StepPreprocess, self).__init__()

    self._hparams = hparams
    if self._hparams.recurrence_type == "act":
      num_steps = self._hparams.act_max_steps
    else:
      num_steps = self._hparams.num_rec_steps

    channels = self._hparams.hidden_size

    self.get_layer_timing_signal_learned_1d = common_attention.GetLayerTimingSignalLearned1D(
      channels, num_steps
    )

    print('Test')

  def call(self, inputs, step, training=None, mask=None):
    if self._hparams.add_position_timing_signal:
      inputs = self.add_position_timing_signal(inputs, self._hparams)

    if self._hparams.add_step_timing_signal:
      inputs = self.add_step_timing_signal(inputs, step)

    return inputs

  def add_step_timing_signal(self, x, step):

    if self._hparams.step_timing_signal_type == "learned":
      signal = self.get_layer_timing_signal_learned_1d(step)
    else:
      raise ValueError("Unknown step_timing_signal_type")

    if self._hparams.add_or_concat_timing_signal == "add":
      x_with_timing = x + common_layers.cast_like(signal, x)
    else:
      raise ValueError("Unknown add_or_concat_timing_signal")

    return x_with_timing

  @staticmethod
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
      return x_with_timing

    else:
      ValueError("Unknown timing signal add or concat type: %s"
                 % hparams.add_or_concat_timing_signal)
