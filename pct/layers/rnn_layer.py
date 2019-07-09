# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from pct.layers import common_layers


class ElmanRNNCell(tf.keras.Model):
  """ Elman Recurrent Neural Network Cell

  Paper: https://www.cs.swarthmore.edu/~meeden/cs63/f07/elman.srn.pdf
  This class represents the most basic cell of recurrent neural network. For later use, output
  units aren't added to it.

  Arguments:
    input_dim
    hidden_dim
    output_dim
  """

  def __init__(self, input_dim, hidden_dim, output_dim):

    super(ElmanRNNCell, self).__init__()
    self._weight_ih = tf.get_variable(
      "elman_weight_ih", [input_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
    )
    self._weight_hh = tf.get_variable(
      "elman_weight_hh", [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer(),
    )
    self._hidden_bias = tf.get_variable(
      "elman_hidden_bias", [1, hidden_dim], initializer=tf.zeros_initializer()
    )
    self._weight_hy = tf.get_variable(
      "elman_weight_hy", [hidden_dim, output_dim], initializer=tf.contrib.layers.xavier_initializer(),
    )
    self._output_bias = tf.get_variable(
      "elman_output_bias", [1, output_dim], initializer=tf.zeros_initializer()
    )

  def call(self, x, hidden_layer_previous, training=False, mask=None):
    hidden_layer_current = tf.math.tanh(
      tf.matmul(x, self._weight_ih) + tf.matmul(hidden_layer_previous, self._weight_hh) + self._hidden_bias
    )
    output_current = tf.math.tanh(
      tf.matmul(hidden_layer_current, self._weight_hy) + self._output_bias
    )

    return hidden_layer_current, output_current


class ACTChunkingRNNLayer(tf.keras.Model):

  def __init__(self, cell_type, act_halting_bias_init, act_epsilon, *cell_args):
    super(ACTChunkingRNNLayer, self).__init__()
    self._act_epsilon = act_epsilon
    self.rnn_cell = cell_type(*cell_args)
    self.halting_unit = common_layers.Dense(
      1,
      activation=tf.nn.sigmoid,
      use_bias=True,
      bias_initializer=tf.constant_initializer(
        act_halting_bias_init
      )
    )

  def call(self, features, initial_state, training=False, mask=None):
    intermediate_states = []
    outputs = []

    state = tf.identity(initial_state)
    feature_mask = tf.not_equal(tf.reduce_sum(tf.abs(features), axis=-1), 0)
    feature_mask = tf.cast(feature_mask, tf.float32)
    for i in range(features.shape[1]):
      out, state = self.rnn_cell(features[:, i, :], state)

      new_output_flag = tf.cast(
        tf.greater(self.halting_unit(state), 1.0 - self._act_epsilon),
        tf.float32
      )

      # update outputs based on the halting probability
      outputs.append(new_output_flag * out * feature_mask[:, i, tf.newaxis])
      # update states based on the halting probability
      # if there is output, cut the connection between current state and next state
      state = (1 - new_output_flag) * state * feature_mask[:, i, tf.newaxis]

      intermediate_states.append(state)

    return tf.stack(outputs), tf.stack(intermediate_states)
