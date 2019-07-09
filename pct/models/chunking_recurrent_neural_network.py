# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.utils import registry
from pct.utils import pretrain_model
from pct.utils import hparams_lib
from pct.layers import common_layers
from pct.layers import common_attention
from pct.layers import rnn_layer
from pct.models import universal_transformer_util


@registry.register_model
class ChunkingElmanRNN(pretrain_model.PretrainModel):

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               **kwargs):
    super(ChunkingElmanRNN, self).__init__(hparams, mode, problem_hparams, **kwargs)
    self.elman_rnn_layer = rnn_layer.ACTChunkingRNNLayer(rnn_layer.ElmanRNNCell,
                                                         hparams.act_halting_bias_init,
                                                         hparams.act_epsilon,
                                                         hparams.hidden_size,
                                                         hparams.hidden_size,
                                                         hparams.hidden_size)

  def body(self, features, training):
    inputs = features["inputs"]
    inputs = common_layers.flatten4d3d(inputs)

    zero_initial_states = tf.zeros([inputs.shape[0], self._hparams.hidden_size])
    outputs, intermediate_states = self.elman_rnn_layer(inputs, zero_initial_states, training=training)

    return outputs, intermediate_states
