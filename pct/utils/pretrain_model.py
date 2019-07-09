# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections

from pct.utils import hparams_lib
from pct.utils.base_model import average_sharded_losses
from pct.layers import modalities
from pct.layers import common_layers


class PretrainModel(tf.keras.Model):

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               **kwargs):
    super(PretrainModel, self).__init__(**kwargs)

    # setup hparams
    self._problem_hparams = problem_hparams
    self._original_hparams = hparams_lib.copy_hparams(hparams)
    self.set_mode(mode)

    self.symbol_bottom_inputs = modalities.SymbolBottomSimple(
      self._original_hparams,
      self._problem_hparams.vocab_size["inputs"],
    )

    self.predict_dense = common_layers.DenseReluDense(
      self._hparams.filter_size,
      self._hparams.hidden_size,
    )

  def call(self, inputs, training=False, **kwargs):

    features = inputs

    if self._original_hparams.pretrain_type == "autoregressive":
      if self._hparams.mode in [tf.estimator.ModeKeys.TRAIN]:
        autoregressive_features, autoregressive_targets = self.autoregressive_train_prepare(features)
      else:
        autoregressive_features = {"inputs": tf.identity(features["inputs"])}
      transformed_features = self.bottom(autoregressive_features, training)
      target_features, losses_dict = self.autoregressive_predict(transformed_features, training)
      logits = self.top(target_features)

      if self._hparams.mode in [tf.estimator.ModeKeys.TRAIN]:
        losses_dict['training'] = self.loss(logits, autoregressive_targets["inputs"])
        losses_dict = average_sharded_losses(losses_dict)

    else:
      raise ValueError("Unknown pretrain_type in original hyperparameters.")

    loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))
    return logits, loss

  def bottom(self, features, training):

    transformed_features = collections.OrderedDict()

    # Transforming input features
    feature_name = "inputs"
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    tf.logging.info("Transforming feature '%s' with %s.bottom",
                 feature_name,
                 modality_name)
    transformed_features[feature_name] = self.symbol_bottom_inputs(features[feature_name], training)
    transformed_features[feature_name + "_raw"] = features[feature_name]

    return transformed_features

  def top(self, output):
    feature_name = "inputs"
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    tf.logging.info("Transforming body output with %s.top", modality_name)
    logits = modalities.symbol_top(output, self.symbol_bottom_inputs.embedding_space)

    return logits

  def autoregressive_predict(self, features, training):

    outputs, intermediate_states = self.body(features, training)

    # brutal addition to create final outputs (abstract representation of the previous sequence)
    # sum across word dimension
    transformed_outputs = tf.reduce_sum(outputs, axis=0)

    intermediate_states = tf.transpose(intermediate_states, [1, 0, 2])
    intermediate_mask_matrix = tf.not_equal(tf.reduce_sum(tf.abs(tf.squeeze(features["inputs"])), axis=-1), 0)
    intermediate_mask_matrix_shift = intermediate_mask_matrix[:, 1:]
    intermediate_mask_matrix_shift = tf.pad(
      intermediate_mask_matrix_shift,
      [[0, 0], [0, 1]],
      'CONSTANT',
      constant_values=False)
    edge_detector = tf.cast(intermediate_mask_matrix, tf.float32) - \
                    tf.cast(intermediate_mask_matrix_shift, tf.float32)
    edge_index = tf.where(tf.equal(edge_detector, 1))
    transformed_intermediate_states = tf.gather_nd(intermediate_states, edge_index)

    # # select specific bands of intermediate_states
    # selected_intermediate_states = tf.transpose(intermediate_states, [2, 1, 0])
    # selected_intermediate_states = tf.linalg.band_part(selected_intermediate_states, 0, 0)
    # selected_intermediate_states = tf.transpose(selected_intermediate_states, [1, 2, 0])
    # transformed_intermediate_states = tf.reduce_sum(selected_intermediate_states, axis=1)

    concatenate_autoregressive_features = tf.concat([transformed_outputs, transformed_intermediate_states], axis=1)

    # predict next word
    predict_word_representations = self.predict_dense(
      concatenate_autoregressive_features,
      training,
    )

    return predict_word_representations[:, tf.newaxis, tf.newaxis, :], {}

  @staticmethod
  def autoregressive_train_prepare(features):
    assert len(features["inputs"].shape) == 4
    # assert features["inputs"].shape[0] == 1

    autoregressive_features = None
    autoregressive_targets = None
    for item_index in range(features["inputs"].shape[0]):
      # delete the case of not having any words
      processed_tmp_features = tf.squeeze(features["inputs"][item_index])
      processed_tmp_features = tf.tile(
        tf.expand_dims(processed_tmp_features, axis=0),
        [processed_tmp_features.shape[0], 1]
      )

      # generate autoregressive prediction features
      autoregressive_tmp_features = tf.matrix_band_part(processed_tmp_features, -1, 0)
      autoregressive_tmp_features = autoregressive_tmp_features[:-1, :]
      autoregressive_tmp_features = tf.expand_dims(autoregressive_tmp_features, 2)
      autoregressive_tmp_features = tf.expand_dims(autoregressive_tmp_features, 3)

      # generate autoregressive prediction targets
      autoregressive_tmp_targets = tf.matrix_band_part(processed_tmp_features, 0, 1) - tf.matrix_band_part(processed_tmp_features, 0, 0)
      autoregressive_tmp_targets = tf.reduce_sum(autoregressive_tmp_targets, axis=1)
      autoregressive_tmp_targets = autoregressive_tmp_targets[:-1]
      autoregressive_tmp_targets = autoregressive_tmp_targets[:, tf.newaxis, tf.newaxis, tf.newaxis]

      if autoregressive_features is None:
        autoregressive_features = tf.identity(autoregressive_tmp_features)
      else:
        autoregressive_features = tf.concat(
          [autoregressive_features, autoregressive_tmp_features],
          axis=0
        )

      if autoregressive_targets is None:
        autoregressive_targets = tf.identity(autoregressive_tmp_targets)
      else:
        autoregressive_targets = tf.concat(
          [autoregressive_targets, autoregressive_tmp_targets],
          axis=0
        )

    return {"inputs": autoregressive_features}, {"inputs": autoregressive_targets}

  def loss(self, logits, targets):
    loss_num, loss_den = modalities.generic_loss(logits, targets, self._hparams)
    return loss_num, loss_den

  def set_mode(self, mode):
    """Set hparams with the given mode."""
    tf.logging.info("Setting BaseModel mode to '%s'", mode)
    hparams = hparams_lib.copy_hparams(self._original_hparams)
    hparams.add_hparam("mode", mode)
    # When not in training mode, set all forms of dropout to zero.
    if mode != tf.estimator.ModeKeys.TRAIN:
      for key in hparams.values():
        if key.endswith("dropout") or key == "label_smoothing":
          tf.logging.info("Setting hparams.%s to 0.0", key)
          setattr(hparams, key, 0.0)
    self._hparams = hparams

  def optimizer(self):
    """Return a training op minimizing loss."""
    # lr = learning_rate.learning_rate_schedule(self._hparams)
    started_learning_rate = self._hparams.learning_rate / 1e3
    learning_rate = tf.train.exponential_decay(
      started_learning_rate,
      self._hparams.train_steps,
      decay_steps=self._hparams.train_steps/2,
      decay_rate=0.96,
      staircase=True
    )

    optimizer = tf.train.AdamOptimizer(
      learning_rate=learning_rate,
      beta1=self._hparams.optimizer_adam_beta1,
      beta2=self._hparams.optimizer_adam_beta2,
      epsilon=self._hparams.optimizer_adam_epsilon,
    )
    return optimizer
