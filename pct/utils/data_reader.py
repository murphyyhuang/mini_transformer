# coding=utf-8

"""Data reader module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from natsort import natsort
import os


def decode_tfrecord_translation(features):
  decoded_dict = {
    'inputs': tf.VarLenFeature(tf.int64),
    'targets': tf.VarLenFeature(tf.int64)
  }

  decoded_features = tf.parse_single_example(features, decoded_dict)

  output_dict = {
    'inputs': decoded_features['inputs'].values,
    'targets': decoded_features['targets'].values
  }
  return output_dict


def construct_pattern(hparams, data_type):
  assert data_type in ['train', 'dev'], "Invalid data type to construct data file pattern"
  return hparams.problem + '-' + data_type


def get_data_files(hparams, filepattern):
  """
  This function has been realized in tf.contrib.slim.parallel_reader.get_data_files.
  But afraid of modification in tf.contrib, here we define a simple function generating same result.
  :return:
  """
  all_files = os.listdir(hparams.data_dir)
  selected_files = [os.path.join(hparams.data_dir, filename)
                    for filename in all_files if filepattern in filename]
  # sort the name of selected file respecting their filenames
  return natsort.natsorted(selected_files)


def create_generator(hparams, mode):
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  filepattern = construct_pattern(hparams, mode)
  data_files = get_data_files(hparams, filepattern)
  dataset = tf.data.TFRecordDataset(data_files, buffer_size=8 * 1024 * 1024)
  dataset = dataset.map(decode_tfrecord_translation, num_parallel_calls=hparams.num_threads)

  if hparams.shuffle_files and is_training:
    dataset = dataset.shuffle(hparams.shuffle_buffer_size)


  return dataset

