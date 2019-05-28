# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2


flags = tf.flags
FLAGS = flags.FLAGS


def create_run_config():
  pass


def create_experiment_fn():
  pass


def schedule_experiment():
  pass


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  create_run_config()

  create_session_config()

  create_experiment_fn()

  schedule_experiment()
