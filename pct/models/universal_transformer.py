# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.utils import registry
from pct.utils import base_model
from pct.layers import common_layers
from pct.layers import common_attention
from pct.models import universal_transformer_util


@registry.register_model
class UniversalTransformer(base_model.BaseModel):

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               **kwargs):
    super(UniversalTransformer, self).__init__(hparams, mode, problem_hparams, **kwargs)

    self.universal_transformer_encoder = universal_transformer_util.UniversalTransformerEncoder(
      hparams
    )
    self.universal_transformer_decoder = universal_transformer_util.UniversalTransformerDecoder(
      hparams
    )

  def encode(self, encoder_input, training):

    (encoder_output, encoder_extra_output) = (
      self.universal_transformer_encoder(encoder_input, training)
    )

    return encoder_output, encoder_extra_output

  def decode(self, decoder_input, encoder_output, decoder_self_attention_bias, training):

    (decoder_output, dec_extra_output) = (
      self.universal_transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        training
      )
    )

    return decoder_output, dec_extra_output

  def body(self, features, training):
    """Universal Transformer main model_fn.


    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "targets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams
    if hparams.add_position_timing_signal:
      # Turning off addition of positional embedding in the encoder/decoder
      # preparation as we do it in the beginning of each step.
      hparams.pos = None

    inputs = features["inputs"]
    inputs = common_layers.flatten4d3d(inputs)
    (encoder_output, enc_extra_output) = self.encode(inputs, training)
    # if self.has_input:
    #   inputs = features["inputs"]
    #   (encoder_output, enc_extra_output) = self.encode(inputs, hparams)
    # else:
    #   (encoder_output, enc_extra_output) = (None, (None, None))

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    (decoder_input,
     decoder_self_attention_bias) = transformer_prepare_decoder(targets)

    decoder_output, dec_extra_output = self.decode(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        training
    )

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
      if self.has_input:
        enc_ponder_times, enc_remainders = enc_extra_output
        enc_act_loss = (
            hparams.act_loss_weight *
            tf.reduce_mean(enc_ponder_times + enc_remainders))
      else:
        enc_act_loss = 0.0

      (dec_ponder_times, dec_remainders) = dec_extra_output
      dec_act_loss = (
          hparams.act_loss_weight *
          tf.reduce_mean(dec_ponder_times + dec_remainders))
      act_loss = enc_act_loss + dec_act_loss
      tf.contrib.summary.scalar("act_loss", act_loss)
      return decoder_output, {"act_loss": act_loss}

    return decoder_output


def transformer_prepare_decoder(targets):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """

  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(
          common_layers.shape_list(targets)[1]))

  decoder_input = common_layers.shift_right_3d(targets)

  return (decoder_input, decoder_self_attention_bias)
