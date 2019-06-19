# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from pct.utils import text_encoder
from pct.utils import hparam

# Enable TF Eager execution
tfe = tf.contrib.eager


def text2text_encoders(hparams):
  source_vocab_filename = os.path.join(hparams.data_dir, hparams.source_vocab_filename)
  target_vocab_filename = os.path.join(hparams.data_dir, hparams.target_vocab_filename)
  source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
  target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)

  # create HParams instance
  hp = hparam.HParams()
  hp.add_hparam("vocab_size", {
    "inputs": source_token.vocab_size,
    "targets": target_token.vocab_size,
  })
  hp.add_hparam("stop_at_eos", text_encoder.EOS_ID)

  return {
    "inputs": source_token,
    "targets": target_token,
    "hparams": hp,
  }


def decode_from_file(encoders, text2text_model, hparams):

  def encode(input_str):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}

  def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
      integers = integers[:integers.index(1)]

    if np.asarray(integers).shape[-1] == 1:
      return encoders["targets"].decode(integers)
    else:
      return encoders["targets"].decode(np.squeeze(integers))

  output_writer = open(hparams.output_dir, 'a+')
  with open(hparams.decode_dir, 'r') as file_reader:
    for sentence in file_reader:
      sentence = sentence.strip()
      encoded_inputs = encode(sentence)
      model_output = text2text_model.infer(encoded_inputs)
      decode_output = decode(model_output)
      output_writer.write(decode_output + '\n')
  output_writer.close()
