# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BaseModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import six
import math
import collections
import tensorflow as tf

from pct.utils import hparams_lib
from pct.utils import optimize
from pct.utils import learning_rate
from pct.layers import common_layers
from pct.layers import modalities

from tensorflow.python.ops import variable_scope


class DummyVariableStore(object):

  @contextlib.contextmanager
  def as_default(self):
    yield


def create_eager_var_store():
  if tf.executing_eagerly():
    return variable_scope.EagerVariableStore()
  else:
    return DummyVariableStore()


def summarize_features(features, num_shards=1):
  """Generate summaries for features."""
  if not common_layers.should_generate_summaries():
    return

  with tf.name_scope("input_stats"):
    for (k, v) in sorted(six.iteritems(features)):
      if (isinstance(v, tf.Tensor) and (v.get_shape().ndims > 1) and (v.dtype != tf.string)):
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


def average_sharded_losses(sharded_losses):
  """Average losses across datashards.

  Args:
    sharded_losses: list<dict<str loss_name, Tensor loss>>. The loss
      can be a single Tensor or a 2-tuple (numerator and denominator).

  Returns:
    losses: dict<str loss_name, Tensor avg_loss>
  """
  losses = {}
  for loss_name in sorted(sharded_losses[0]):
    all_shards = [shard_losses[loss_name] for shard_losses in sharded_losses]
    if isinstance(all_shards[0], tuple):
      sharded_num, sharded_den = zip(*all_shards)
      mean_loss = (
          tf.add_n(sharded_num) / tf.maximum(
              tf.cast(1.0, sharded_den[0].dtype), tf.add_n(sharded_den)))
    else:
      mean_loss = tf.reduce_mean(all_shards)

    losses[loss_name] = mean_loss
  return losses


class BaseModel(tf.keras.Model):

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               **kwargs):
    super(BaseModel, self).__init__(name=hparams.model, **kwargs)

    # setup hparams
    self._problem_hparams = problem_hparams
    self._original_hparams = hparams_lib.copy_hparams(hparams)
    self.set_mode(mode)
    # self._decode_hparams = hparams_lib.copy_hparams(
    #   decode_hparams) if decode_hparams is not None else None

    self._eager_var_store = create_eager_var_store()

  def call(self, inputs, **kwargs):
    features = inputs
    # set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
      optimize.get_variable_initializer(self.hparams))
    with self._eager_var_store.as_default():
      summarize_features(features)
      sharded_logits, losses = self.model_fn(features)
      return tf.concat(sharded_logits, 0), losses

  def bottom(self, features):
    transformed_features = collections.OrderedDict()

    # Transforming input features
    feature_name = 'inputs'
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)
    with tf.variable_scope(modality_name):
      tf.logging.info("Transforming feature '%s' with %s.bottom",
                   feature_name,
                   modality_name)
      transformed_features[feature_name] = modalities.symbol_bottom_simple(
        features[feature_name], self._hparams, vocab_size, "input_emb")
      transformed_features[feature_name + "_raw"] = features[feature_name]

    # Transforming target features
    feature_name = 'targets'
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)
    with tf.variable_scope(modality_name):
      tf.logging.info("Transforming feature '%s' with %s.targets_bottom",
                   feature_name,
                   modality_name)
      transformed_features[feature_name] = modalities.symbol_bottom_simple(
        features[feature_name], self._hparams, vocab_size, "target_emb")
      transformed_features[feature_name + "_raw"] = features[feature_name]

    return transformed_features

  def top(self, output, features):

    del features  # unused

    feature_name = "targets"
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    with tf.variable_scope(modality_name):
      tf.logging.info("Transforming body output with %s.top", modality_name)
      logits = modalities.symbol_top(output, self._hparams, vocab_size)

    return logits

  def loss(self, logits, features):
    loss_num, loss_den = modalities.generic_loss(logits, features, self._hparams)
    return loss_num, loss_den

  def model_fn(self, features):
    transformed_features = self.bottom(features)

    with tf.variable_scope("body"):
      tf.logging.info("Building model body")
      output = self.body(transformed_features)
    losses = output[-1]
    logits = self.top(output, features)

    if self._hparams.mode == tf.estimator.ModeKeys.TRAIN:
      losses['training'] = self.loss(logits, features)

    losses = average_sharded_losses(losses)
    return logits, losses

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
    lr = learning_rate.learning_rate_schedule(self._hparams)
    optimizer = tf.train.AdamOptimizer(
      learning_rate=lr,
      beta1=self._hparams.optimizer_adam_beta1,
      beta2=self._hparams.optimizer_adam_beta2,
      epsilon=self._hparams.optimizer_adam_epsilon,
    )
    return optimizer

  @property
  def hparams(self):
    return self._hparams
