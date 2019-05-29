# coding=utf-8

"""Data reader module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import multiprocessing
import random
import os
import six


_file_num_records_cache = {}


def cpu_count():
  """Return the number of available cores."""
  num_available_cores = multiprocessing.cpu_count()
  return num_available_cores


def _file_num_records_cached(filename):
  """Return the number of TFRecords in a file."""
  # Cache the result, as this is expensive to compute
  if filename in _file_num_records_cache:
    return _file_num_records_cache[filename]
  ret = 0
  for _ in tf.python_io.tf_record_iterator(filename):
    ret += 1
  _file_num_records_cache[filename] = ret
  return ret


def skip_random_fraction(dataset, data_file):
  # Skip a random fraction at the beginning of the stream.  The skip is
  # essential for synchronous highly-parallel training to avoid multiple
  # replicas reading the same data in lock-step.
  num_skip = random.randint(0, _file_num_records_cached(data_file))
  return dataset.skip(num_skip)


def cast_ints_to_int32(features):
  f = {}
  for k, v in sorted(six.iteritems(features)):
    if v.dtype in [tf.int64, tf.uint8]:
      v = tf.to_int32(v)
    f[k] = v
  return f


def example_length(example):
  length = 0
  # Length of the example is the maximum length of the feature lengths
  for _, v in sorted(six.iteritems(example)):
    # For images the sequence length is the size of the spatial dimensions.
    feature_length = tf.shape(v)[0]
    if len(v.get_shape()) > 2:
      feature_length = tf.shape(v)[0] * tf.shape(v)[1]
    length = tf.maximum(length, feature_length)
  return length


def example_valid_size(example, min_length, max_length):
  length = example_length(example)
  return tf.logical_and(
      length >= min_length,
      length <= max_length,
  )


def standardize_shapes(features, batch_size=None):
  """Set the right shapes for the features."""
  for fname in ["inputs", "targets"]:
    if fname not in features:
      continue
    f = features[fname]
    while len(f.get_shape()) < 4:
      f = tf.expand_dims(f, axis=-1)
    features[fname] = f

  if batch_size:
    # Ensure batch size is set on all features
    for _, t in six.iteritems(features):
      shape = t.get_shape().as_list()
      shape[0] = batch_size
      t.set_shape(t.get_shape().merge_with(shape))
      # Assert shapes are fully known
      t.get_shape().assert_is_fully_defined()

  return features


def pad_batch(features, batch_multiple):
  """Pad batch dim of features to nearest multiple of batch_multiple."""
  feature = list(features.items())[0][1]
  batch_size = tf.shape(feature)[0]
  mod = batch_size % batch_multiple
  has_mod = tf.cast(tf.cast(mod, tf.bool), tf.int32)
  batch_padding = batch_multiple * has_mod - mod

  padded_features = {}
  for k, feature in features.items():
    rank = len(feature.shape)
    paddings = [[0, 0] for _ in range(rank)]
    paddings[0][1] = batch_padding
    padded_feature = tf.pad(feature, paddings)
    padded_features[k] = padded_feature
  return padded_features


def _summarize_features(features, num_shards=1):
  with tf.name_scope("input_stats"):
    for (k, v) in six.iteritems(features):
      if isinstance(v, tf.Tensor) and v.get_shape().ndims > 1:
        tf.summary.scalar("%s_batch" % k, tf.shape(v)[0] // num_shards)
        tf.summary.scalar("%s_length" % k, tf.shape(v)[1])
        nonpadding = tf.to_float(tf.not_equal(v, 0))
        nonpadding_tokens = tf.reduce_sum(nonpadding)
        tf.summary.scalar("%s_nonpadding_tokens" % k, nonpadding_tokens)
        tf.summary.scalar("%s_nonpadding_fraction" % k,
                          tf.reduce_mean(nonpadding))


def _bucket_boundaries(max_length, min_length=8, length_bucket_step=1.1):
  """A default set of length-bucket boundaries."""
  assert length_bucket_step > 1.0
  x = min_length
  boundaries = []
  while x < max_length:
    boundaries.append(x)
    x = max(x + 1, int(x * length_bucket_step))
  return boundaries


def batching_scheme(batch_size,
                    max_length,
                    min_length_bucket,
                    length_bucket_step,
                    drop_long_sequences=False,
                    shard_multiplier=1,
                    length_multiplier=1,
                    min_length=0):
  """A batching scheme based on model hyperparameters.

  Every batch contains a number of sequences divisible by `shard_multiplier`.

  Args:
    batch_size: int, total number of tokens in a batch.
    max_length: int, sequences longer than this will be skipped. Defaults to
      batch_size.
    min_length_bucket: int
    length_bucket_step: float greater than 1.0
    drop_long_sequences: bool, if True, then sequences longer than
      `max_length` are dropped.  This prevents generating batches with
      more than the usual number of tokens, which can cause out-of-memory
      errors.
    shard_multiplier: an integer increasing the batch_size to suit splitting
      across datashards.
    length_multiplier: an integer multiplier that is used to increase the
      batch sizes and sequence length tolerance.
    min_length: int, sequences shorter than this will be skipped.

  Returns:
     A dictionary with parameters that can be passed to input_pipeline:
       * boundaries: list of bucket boundaries
       * batch_sizes: list of batch sizes for each length bucket
       * max_length: int, maximum length of an example

  Raises:
    ValueError: If min_length > max_length
  """
  max_length = max_length or batch_size
  if max_length < min_length:
    raise ValueError("max_length must be greater or equal to min_length")

  boundaries = _bucket_boundaries(max_length, min_length_bucket,
                                  length_bucket_step)
  boundaries = [boundary * length_multiplier for boundary in boundaries]
  max_length *= length_multiplier

  batch_sizes = [
      max(1, batch_size // length) for length in boundaries + [max_length]
  ]
  max_batch_size = max(batch_sizes)
  # Since the Datasets API only allows a single constant for window_size,
  # and it needs divide all bucket_batch_sizes, we pick a highly-composite
  # window size and then round down all batch sizes to divisors of that window
  # size, so that a window can always be divided evenly into batches.
  # TODO(noam): remove this when Dataset API improves.
  highly_composite_numbers = [
      1, 2, 4, 6, 12, 24, 36, 48, 60, 120, 180, 240, 360, 720, 840, 1260, 1680,
      2520, 5040, 7560, 10080, 15120, 20160, 25200, 27720, 45360, 50400, 55440,
      83160, 110880, 166320, 221760, 277200, 332640, 498960, 554400, 665280,
      720720, 1081080, 1441440, 2162160, 2882880, 3603600, 4324320, 6486480,
      7207200, 8648640, 10810800, 14414400, 17297280, 21621600, 32432400,
      36756720, 43243200, 61261200, 73513440, 110270160
  ]
  window_size = max(
      [i for i in highly_composite_numbers if i <= 3 * max_batch_size])
  divisors = [i for i in range(1, window_size + 1) if window_size % i == 0]
  batch_sizes = [max([d for d in divisors if d <= bs]) for bs in batch_sizes]
  window_size *= shard_multiplier
  batch_sizes = [bs * shard_multiplier for bs in batch_sizes]
  # The Datasets API splits one window into multiple batches, which
  # produces runs of many consecutive batches of the same size.  This
  # is bad for training.  To solve this, we will shuffle the batches
  # using a queue which must be several times as large as the maximum
  # number of batches per window.
  max_batches_per_window = window_size // min(batch_sizes)
  shuffle_queue_size = max_batches_per_window * 3

  ret = {
      "boundaries": boundaries,
      "batch_sizes": batch_sizes,
      "min_length": min_length,
      "max_length": (max_length if drop_long_sequences else 10**9),
      "shuffle_queue_size": shuffle_queue_size,
  }
  return ret


def hparams_to_batching_scheme(hparams,
                               drop_long_sequences=False,
                               shard_multiplier=1,
                               length_multiplier=1):
  """Wrapper around _batching_scheme with hparams."""
  return batching_scheme(
      batch_size=hparams.batch_size,
      min_length=hparams.min_length,
      max_length=hparams.max_length,
      min_length_bucket=hparams.min_length_bucket,
      length_bucket_step=hparams.length_bucket_step,
      drop_long_sequences=drop_long_sequences,
      shard_multiplier=shard_multiplier,
      length_multiplier=length_multiplier)


class TextDataGenerator(object):

  def __init__(self, problem_name, mode, hparams, config):

    self.name = problem_name
    self.input_fn = None
    self.input_fn_init(mode, hparams, config)

  def input_fn_init(self,
                    mode,
                    hparams,
                    config,
                    skip_random_fraction_when_training=False):

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    num_threads = cpu_count() if is_training else 1
    dataset_kwargs = {}
    # hparams should contain data_dir
    assert hasattr(hparams, 'data_dir'), "HParams loses the attribute data_dir."
    filepattern = self.filepattern(hparams.data_dir, mode)
    data_files = sorted(tf.data.Dataset.list_files(filepattern))
    dataset_kwargs.update({
        "mode": mode,
        "num_threads": num_threads,
        "hparams": hparams,
        "data_files": data_files
    })
    dataset = self.dataset(**dataset_kwargs)

    if is_training:
      dataset = dataset.repeat()

    if is_training and skip_random_fraction_when_training:
      #  In continuous_train_and_eval when switching between train and
      #  eval, this input_fn method gets called multiple times and it
      #  would give you the exact same samples from the last call
      #  (because the Graph seed is set). So this skip gives you some
      #  shuffling.
      dataset = skip_random_fraction(dataset, data_files[0])

    dataset = dataset.map(cast_ints_to_int32, num_parallel_calls=num_threads)

    def gpu_valid_size(example):
      drop_long_sequences = is_training or hparams.eval_drop_long_sequences
      max_length = self.max_length(hparams)
      max_validate_length = max_length if drop_long_sequences else 10 ** 9
      return example_valid_size(example, hparams.min_length, max_validate_length)

    def define_shapes(example):
      # batch_size = config and config.use_tpu and params["batch_size"]
      # return standardize_shapes(example, batch_size=batch_size)
      return standardize_shapes(example)

    # On GPU, bucket by length
    dataset = dataset.filter(gpu_valid_size)
    cur_batching_scheme = hparams_to_batching_scheme(hparams)

    dataset = dataset.apply(
      tf.data.experimental.bucket_by_sequence_length(
        example_length, cur_batching_scheme["boundaries"],
        cur_batching_scheme["batch_sizes"]))

    if not is_training:
      # TODO(Murphy): I don't want to involve parallel computing in this very early stage. Manually set the num_shards to be 1.
      batch_multiple = 1
      if batch_multiple > 1:
        tf.logging.warn(
          "Padding the batch to ensure that remainder eval batches have "
          "a batch size divisible by the number of data shards. This may "
          "lead to incorrect metrics for non-zero-padded features, e.g. "
          "images. Use a single datashard (i.e. 1 GPU) in that case.")
        dataset = dataset.map(
          functools.partial(pad_batch, batch_multiple=batch_multiple),
          num_parallel_calls=num_threads)

    dataset = dataset.map(define_shapes, num_parallel_calls=num_threads)

    if (is_training and hasattr(hparams, "batch_shuffle_size") and
        hparams.batch_shuffle_size):
      dataset = dataset.shuffle(hparams.batch_shuffle_size)

    def prepare_for_output(example):
      if not config or not config.use_tpu:
        _summarize_features(example)
      if mode == tf.estimator.ModeKeys.PREDICT:
        example["infer_targets"] = example.pop("targets")
        return example
      else:
        return example, example["targets"]

    dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(2)

    return dataset


  def dataset(self,
              mode,
              shuffle_files=None,
              num_threads=None,
              shuffle_buffer_size=1024,
              data_files=None):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    shuffle_files = shuffle_files or shuffle_files is None and is_training

    def _load_records_and_preprocess(filenames):
      """Reads files from a string tensor or a dataset of filenames."""
      # Load records from file(s) with an 8MiB read buffer.
      dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
      dataset = dataset.map(self.decode_example, num_parallel_calls=num_threads)

      return dataset

    if shuffle_files:
      random.shuffle(data_files)

    dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
    if shuffle_files:
      dataset = dataset.apply(tf.data.experimental.parallel_interleave(
        _load_records_and_preprocess, sloppy=True, cycle_length=8))

    if shuffle_files and is_training:
      dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset

  def filepattern(self, data_dir, mode, shard=None):
    """Get filepattern for data files for mode.

    Matches mode to a suffix.
    * DatasetSplit.TRAIN: train
    * DatasetSplit.EVAL: dev
    * DatasetSplit.TEST: test
    * tf.estimator.ModeKeys.PREDICT: dev

    Args:
      data_dir: str, data directory.
      mode: DatasetSplit
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      filepattern str
    """
    path = os.path.join(data_dir, self.name)
    shard_str = "-%05d" % shard if shard is not None else ""
    if mode == DatasetSplit.TRAIN:
      suffix = "train"
    elif mode in [DatasetSplit.EVAL, tf.estimator.ModeKeys.PREDICT]:
      suffix = "dev"
    else:
      assert mode == DatasetSplit.TEST
      suffix = "test"

    return "%s-%s%s*" % (path, suffix, shard_str)

  def decode_example(self, serialized_example):
    decoded_dict = {
      'inputs': tf.VarLenFeature(tf.int64),
      'targets': tf.VarLenFeature(tf.int64)
    }

    decoded_features = tf.parse_single_example(serialized_example, decoded_dict)

    output_dict = {
      'inputs': decoded_features['inputs'].values,
      'targets': decoded_features['targets'].values
    }
    return output_dict

  def max_length(self, model_hparams):
    """Maximum sequence length.

    Problems with fixed length should override.

    Args:
      model_hparams: model hyperparameters
    Returns:
      an integer
    """
    return (model_hparams.split_to_length or model_hparams.max_length or
            model_hparams.batch_size)


class DatasetSplit(object):
  TRAIN = tf.estimator.ModeKeys.TRAIN
  EVAL = tf.estimator.ModeKeys.EVAL
  TEST = "test"
