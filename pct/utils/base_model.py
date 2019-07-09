# coding=utf-8
"""BaseModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import six
import collections
import tensorflow as tf

from pct import global_config
from pct.utils import hparams_lib
from pct.utils import text_encoder
from pct.layers import common_layers
from pct.layers import modalities


class DummyVariableStore(object):

  @contextlib.contextmanager
  def as_default(self):
    yield


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
  for loss_name in sorted(sharded_losses):
    all_shards = sharded_losses[loss_name]
    if isinstance(all_shards, tuple):
      sharded_num, sharded_den = all_shards
      mean_loss = sharded_num / sharded_den
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
    super(BaseModel, self).__init__(**kwargs)

    # setup hparams
    self._problem_hparams = problem_hparams
    self._original_hparams = hparams_lib.copy_hparams(hparams)
    self.set_mode(mode)

    self.symbol_bottom_inputs = modalities.SymbolBottomSimple(
      self._original_hparams,
      self._problem_hparams.vocab_size["inputs"],
    )

    self.symbol_bottom_targets = modalities.SymbolBottomSimple(
      self._original_hparams,
      self._problem_hparams.vocab_size["targets"],
    )

  def call(self, inputs, training=False, **kwargs):
    features = inputs

    summarize_features(features)
    logits, losses_dict = self.model_fn(features, training)

    # sum up different kinds of loss
    loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))
    return logits, loss

  def bottom(self, features, training):

    transformed_features = collections.OrderedDict()

    # Transforming input features
    feature_name = 'inputs'
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    tf.logging.info("Transforming feature '%s' with %s.bottom",
                 feature_name,
                 modality_name)
    transformed_features[feature_name] = self.symbol_bottom_inputs(features[feature_name], training)
    transformed_features[feature_name + "_raw"] = features[feature_name]

    # Transforming target features
    feature_name = 'targets'
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    tf.logging.info("Transforming feature '%s' with %s.targets_bottom",
                 feature_name,
                 modality_name)
    transformed_features[feature_name] = self.symbol_bottom_targets(features[feature_name], training)
    transformed_features[feature_name + "_raw"] = features[feature_name]

    return transformed_features

  def top(self, output, features):

    del features  # unused

    feature_name = "targets"
    vocab_size = self._problem_hparams.vocab_size[feature_name]
    modality_name = "symbol_modality_%d_%d" % (vocab_size, self._hparams.hidden_size)

    tf.logging.info("Transforming body output with %s.top", modality_name)
    logits = modalities.symbol_top(output, self.symbol_bottom_targets.embedding_space)

    return logits

  def loss(self, logits, features):
    loss_num, loss_den = modalities.generic_loss(logits, features['targets'], self._hparams)
    return loss_num, loss_den

  def model_fn(self, features, training):
    transformed_features = self.bottom(features, training)

    tf.logging.info("Building model body")
    body_out = self.body(transformed_features, training)
    # losses = output[-1]
    output, losses = self._normalize_body_output(body_out)
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

  def infer(self, features):
    if self._hparams.beam_size == 1:
      tf.logging.info("Greedy Decoding")
      result = self._greedy_infer(features)
    else:
      raise NotImplementedError
    return result

  def _greedy_infer(self, features):

    features["inputs"] = tf.expand_dims(features["inputs"], 2)
    batch_size = common_layers.shape_list(features["inputs"])[0]
    initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)

    prefix_length = common_layers.shape_list(features["inputs"])[1]
    decode_length = prefix_length + self._hparams.decode_length
    result = initial_output

    vocab_size = self._problem_hparams.vocab_size["targets"]

    logits = tf.zeros((batch_size, 0, 1, 1, vocab_size))
    logits_shape_inv = [None, None, None, None, None]

    loss = 0.0

    def infer_step(recent_output, recent_logits, unuserd_loss):
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      samples, logits, loss = self.sample(features)

      cur_sample = samples[:, -1, :, :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)

      logits = tf.concat([recent_logits, logits[:, -1:]], 1)

      return samples, logits, loss


    def while_exit_cond(result, logits, loss):  # pylint: disable=unused-argument
      """Exit the loop either if reach decode_length or EOS."""
      length = common_layers.shape_list(result)[1]

      not_overflow = length < decode_length

      if self._problem_hparams.stop_at_eos:

        def fn_not_eos():
          return tf.not_equal(  # Check if the last predicted element is a EOS
              tf.squeeze(result[:, -1, :, :]), text_encoder.EOS_ID)

        not_eos = tf.cond(
            # We only check for early stopping if there is at least 1 element (
            # otherwise not_eos will crash).
            tf.not_equal(length, 0),
            fn_not_eos,
            lambda: True,
        )

        return tf.cond(
            tf.equal(batch_size, 1),
            # If batch_size == 1, we check EOS for early stopping.
            lambda: tf.logical_and(not_overflow, not_eos),
            # Else, just wait for max length
            lambda: not_overflow)
      return not_overflow

    result, logits, loss = tf.while_loop(
        while_exit_cond,
        infer_step, [result, logits, loss],
        shape_invariants=[
            tf.TensorShape([None, None, None, None]),
            tf.TensorShape(logits_shape_inv),
            tf.TensorShape([]),
        ],
        back_prop=False,
        parallel_iterations=1,
        name='',
    )

    return result

  def sample(self, features):
    logits, loss = self(features, training=False)
    if self._hparams.sampling_method == "argmax":
      samples = tf.argmax(logits, axis=-1)
    else:
      raise ValueError

    return samples, logits, loss

  @staticmethod
  def _normalize_body_output(body_out):
    if isinstance(body_out, tuple):
      output, losses = body_out
      if isinstance(losses, (list, tuple)):
        losses = {"extra": tf.add_n([tf.reduce_mean(l) for l in losses])}
      elif isinstance(losses, dict):
        pass
      else:
        losses = {"extra": tf.reduce_mean(losses)}
    else:
      output = body_out
      losses = {"extra": 0.0}

    return output, losses

  @property
  def hparams(self):
    return self._hparams
