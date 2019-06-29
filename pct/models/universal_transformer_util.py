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

    self._hparams = hparams
    self.step_preprocess = StepPreprocess(self._hparams)
    self.transformer_encoder_attention_unit = TransformerEncoderAttentionUnit(self._hparams)
    self.transformer_encoder_ffn_unit = TransformerFFNUnit(self._hparams)
    self.encoder_normalizer = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )

    if self._hparams.recurrence_type == "act":
      self.halting_unit = common_layers.Dense(
        1,
        activation=tf.nn.sigmoid,
        use_bias=True,
        bias_initializer=tf.constant_initializer(
          hparams.act_halting_bias_init)
      )

  def call(self, x, training=False, mask=None):

    if self._hparams.recurrence_type == "basic":
      ut_initializer = (x, x, x)  # (state, input, memory)
      ut_encoder_unit = functools.partial(
        universal_transformer_basic,
        encoder_output=None,
        decoder_self_attention_bias=None,
        step_preprocess=self.step_preprocess,
        attention_unit=self.transformer_encoder_attention_unit,
        ffn_unit=self.transformer_encoder_ffn_unit,
        training=training,
      )
      output, _, extra_output = tf.foldl(
        ut_encoder_unit, tf.range(self._hparams.num_rec_steps),
        initializer=ut_initializer,
      )
      output = self.encoder_normalizer(None, output, training)
      return output, extra_output

    elif self._hparams.recurrence_type == "act":
      output, extra_output = universal_transformer_act(x, None, None, self.step_preprocess,
                                                       self.transformer_encoder_attention_unit,
                                                       self.transformer_encoder_ffn_unit,
                                                       self.halting_unit, self._hparams, training)

      output = self.encoder_normalizer(None, output, training)
      return output, extra_output


class UniversalTransformerDecoder(tf.keras.Model):

  def __init__(self, hparams):
    super(UniversalTransformerDecoder, self).__init__()
    self._hparams = hparams

    self.step_preprocess = StepPreprocess(self._hparams)

    self.transformer_decoder_attention_unit = TransformerDecoderAttentionUnit(self._hparams)
    self.transformer_decoder_ffn_unit = TransformerFFNUnit(self._hparams)

    self.decoder_normalizer = common_layers.LayerPrepostprocess(
      'pre',
      self._hparams
    )

    if self._hparams.recurrence_type == "act":
      self.halting_unit = common_layers.Dense(
        1,
        activation=tf.nn.sigmoid,
        use_bias=True,
        bias_initializer=tf.constant_initializer(
          hparams.act_halting_bias_init)
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
        universal_transformer_basic,
        encoder_output=encoder_output,
        decoder_self_attention_bias=decoder_self_attention_bias,
        step_preprocess=self.step_preprocess,
        attention_unit=self.transformer_decoder_attention_unit,
        ffn_unit=self.transformer_decoder_ffn_unit,
        training=training,
      )
      output, _, extra_output = tf.foldl(
        ut_decoder_unit, tf.range(self._hparams.num_rec_steps),
        initializer=ut_initializer,
      )
      output = self.decoder_normalizer(None, output, training)
      return output, extra_output

    if self._hparams.recurrence_type == "act":
      output, extra_output = universal_transformer_act(decoder_input, encoder_output,
                                                       decoder_self_attention_bias, self.step_preprocess,
                                                       self.transformer_decoder_attention_unit,
                                                       self.transformer_decoder_ffn_unit,
                                                       self._hparams, training)

      output = self.decoder_normalizer(None, output, training)
      return output, extra_output


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

  def call(self, inputs, step, training=False, mask=None):
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


class TransformerEncoderAttentionUnit(tf.keras.Model):

  def __init__(self, hparams):
    super(TransformerEncoderAttentionUnit, self).__init__()

    self._hparams = hparams
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

  def call(self, layer_input, encoder_output, decoder_self_attention_bias, training):

    del encoder_output
    new_state_y = self.attention_unit_preprocess(None, layer_input, training)
    new_state_y = self.attention_unit(new_state_y, None, decoder_self_attention_bias, training)
    new_state = self.attention_unit_postprocess(layer_input, new_state_y, training)

    return new_state


class TransformerDecoderAttentionUnit(tf.keras.Model):

  def __init__(self, hparams):
    super(TransformerDecoderAttentionUnit, self).__init__()

    self._hparams = hparams
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

  def call(self, layer_input, encoder_output, decoder_self_attention_bias, training):

    new_state_y = self.attention_unit_preprocess(None, layer_input, training)
    new_state_y = self.attention_unit(new_state_y, None, decoder_self_attention_bias, training)
    new_state = self.attention_unit_postprocess(layer_input, new_state_y, training)
    new_state_y = self.ende_attention_unit_preprocess(None, new_state, training)
    new_state_y = self.ende_attention_unit(new_state_y, encoder_output, None, training)
    new_state = self.ende_attention_unit_postprocess(new_state, new_state_y, training)

    return new_state


class TransformerFFNUnit(tf.keras.Model):

  def __init__(self, hparams):
    super(TransformerFFNUnit, self).__init__()

    self._hparams = hparams

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

  def call(self, layer_input, training):
    new_state_y = self.ffn_unit_preprocess(None, layer_input, training)
    new_state_y = self.ffn_unit(new_state_y, training)
    new_state = self.ffn_unit_postprocess(layer_input, new_state_y, training)

    return new_state


def universal_transformer_basic(layer_inputs,
                                step,
                                encoder_output,
                                decoder_self_attention_bias,
                                step_preprocess,
                                attention_unit,
                                ffn_unit,
                                training):

  state, inputs, memory = tf.unstack(layer_inputs, num=None, axis=0)
  new_state = step_preprocess(state, step)

  new_state = attention_unit(new_state, encoder_output, decoder_self_attention_bias, training)
  new_state = ffn_unit(new_state, training)

  return new_state, inputs, memory


def universal_transformer_act(layer_inputs,
                              encoder_output,
                              decoder_self_attention_bias,
                              step_preprocess,
                              attention_unit,
                              ffn_unit,
                              halting_unit,
                              hparams,
                              training):
  state = layer_inputs
  act_max_steps = hparams.act_max_steps
  threshold = 1.0 - hparams.act_epsilon
  state_shape_static = state.get_shape()

  state_slice = slice(0, 2)

  # Dynamic shape for update tensors below
  update_shape = tf.shape(state)[state_slice]

  # Halting probabilities (p_t^n in the paper)
  halting_probability = tf.zeros(update_shape)

  # Remainders (R(t) in the paper)
  remainders = tf.zeros(update_shape)

  # Number of updates performed (N(t) in the paper)
  n_updates = tf.zeros(update_shape)

  # Previous cell states (s_t in the paper)
  previous_state = tf.zeros_like(state)
  step = tf.constant(0, dtype=tf.int32)

  def ut_function(state, step, halting_probability, remainders, n_updates,
                  previous_state):
    """implements act (position-wise halting).

    Args:
      state: 3-D Tensor: [batch_size, length, channel]
      step: indicates number of steps taken so far
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      previous_state: previous state

    Returns:
      transformed_state: transformed state
      step: step+1
      halting_probability: halting probability
      remainders: act remainders
      n_updates: act n_updates
      new_state: new state
    """
    state = step_preprocess(state, step, hparams)

    p = halting_unit(state)
    # maintain position-wise probabilities
    p = tf.squeeze(p, axis=-1)

    # Mask for inputs which have not halted yet
    still_running = tf.cast(tf.less(halting_probability, 1.0), tf.float32)

    # Mask of inputs which halted at this step
    new_halted = tf.cast(
      tf.greater(halting_probability + p * still_running, threshold),
      tf.float32) * still_running

    # Mask of inputs which haven't halted, and didn't halt this step
    still_running = tf.cast(
      tf.less_equal(halting_probability + p * still_running, threshold),
      tf.float32) * still_running

    # Add the halting probability for this step to the halting
    # probabilities for those input which haven't halted yet
    halting_probability += p * still_running

    # Compute remainders for the inputs which halted at this step
    remainders += new_halted * (1 - halting_probability)

    # Add the remainders to those inputs which halted at this step
    halting_probability += new_halted * remainders

    # Increment n_updates for all inputs which are still running
    n_updates += still_running + new_halted

    # Compute the weight to be applied to the new state and output
    # 0 when the input has already halted
    # p when the input hasn't halted yet
    # the remainders when it halted this step
    update_weights = tf.expand_dims(
      p * still_running + new_halted * remainders, -1)

    # apply transformation on the state
    transformed_state = state

    transformed_state = attention_unit(transformed_state, encoder_output, decoder_self_attention_bias, training)
    transformed_state = ffn_unit(transformed_state, training)
    # update running part in the weighted state and keep the rest
    new_state = ((transformed_state * update_weights) +
                 (previous_state * (1 - update_weights)))

    # remind TensorFlow of everything's shape
    transformed_state.set_shape(state_shape_static)
    for x in [halting_probability, remainders, n_updates]:
      x.set_shape(state_shape_static[state_slice])
    new_state.set_shape(state_shape_static)
    step += 1
    return (transformed_state, step, halting_probability, remainders, n_updates,
            new_state)

  # While loop stops when this predicate is FALSE.
  # Ie all (probability < 1-eps AND counter < N) are false.
  def should_continue(u0, u1, halting_probability, u2, n_updates, u3):
    del u0, u1, u2, u3
    return tf.reduce_any(
      tf.logical_and(
        tf.less(halting_probability, threshold),
        tf.less(n_updates, act_max_steps)))

  # Do while loop iterations until predicate above is false.
  (_, _, _, remainder, n_updates, new_state) = tf.while_loop(
    should_continue, ut_function,
    (state, step, halting_probability, remainders, n_updates, previous_state),
    maximum_iterations=act_max_steps + 1)

  ponder_times = n_updates
  remainders = remainder

  tf.contrib.summary.scalar("ponder_times", tf.reduce_mean(ponder_times))

  return new_state, (ponder_times, remainders)
