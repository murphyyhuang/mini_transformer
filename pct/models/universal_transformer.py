# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from pct.models import universal_transformer_util


class UniversalTransformer(object):

  def encode(self, encoder_input, hparams):

    (encoder_output, encoder_extra_output) = (
      universal_transformer_util.universal_transformer_encoder(
        encoder_input,
        hparams
      )
    )

    return encoder_output, encoder_extra_output

  def decode(self, decoder_input, encoder_output, hparams):

    (decoder_output, dec_extra_output) = (
      universal_transformer_util.universal_transformer_decoder(
        decoder_input,
        encoder_output,
        hparams
      )
    )

    return decoder_output, dec_extra_output
