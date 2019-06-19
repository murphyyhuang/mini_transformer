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
    new_state_y = self.attention_unit(new_state_y, None, None, training)
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

  def call(self,
           decoder_input,
           encoder_output,
           decoder_self_attention_bias,
           training=False,
           mask=None):

    if self._hparams.recurrence_type == "basic":
      x = decoder_input
      ut_initializer = (x, x, x)  # (state, input, memory)
      ut_decoder_unit = functools.partial(
        self._universal_transformer_basic,
        encoder_output=encoder_output,
        decoder_self_attention_bias=decoder_self_attention_bias,
        training=training,
      )
      output, _, extra_output = tf.foldl(
        ut_decoder_unit, tf.range(self._hparams.num_rec_steps),
        initializer=ut_initializer,
      )
      output = self.decoder_normalizer(None, output, training)
      return output, extra_output

  def _universal_transformer_basic(self,
                                   decoder_input,
                                   step,
                                   encoder_output,
                                   decoder_self_attention_bias,
                                   training):
    state, inputs, memory = tf.unstack(decoder_input, num=None, axis=0, name="unstack")
    new_state = self.step_preprocess(state, step)

    new_state_y = self.attention_unit_preprocess(None, new_state, training)
    new_state_y = self.attention_unit(new_state_y, None, decoder_self_attention_bias, training)
    new_state = self.attention_unit_postprocess(new_state, new_state_y, training)
    new_state_y = self.ende_attention_unit_preprocess(None, new_state, training)
    new_state_y = self.ende_attention_unit(new_state_y, encoder_output, None, training)
    new_state = self.ende_attention_unit_postprocess(new_state, new_state_y, training)
    new_state_y = self.ffn_unit_preprocess(None, new_state, training)
    new_state_y = self.ffn_unit(new_state_y, training)
    new_state = self.ffn_unit_postprocess(new_state, new_state_y, training)

    return new_state, inputs, memory

  def _universal_transformer_act(self):
    pass


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
