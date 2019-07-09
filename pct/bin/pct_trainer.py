# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pct.utils import data_reader
from pct.utils import hparams_lib
from pct.utils import base_model_lib
from pct.utils import trainer_lib
from pct import global_config

# force registration of models
from pct import models # pylint: disable=unused-import

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_string("config_dir", None, "Directory with training data.")

session_config = trainer_lib.create_session_config()
tf.enable_eager_execution(config=session_config)


def main(argv):
  tf.logging.set_verbosity(global_config.LOGGING_LEVEL)

  # read hyper-parameters
  assert hasattr(FLAGS, 'config_dir'), "'config_dir' must be given to set all the hyper-parameters of this project."
  model_hparams = hparams_lib.create_hparams(FLAGS.config_dir)

  # initialize summary writer
  train_summary_writer = tf.contrib.summary.create_file_writer(
    model_hparams.model_dir,
    flush_millis=10000,
  )

  with train_summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    train_data_generator = data_reader.TextDataGenerator(tf.estimator.ModeKeys.TRAIN, model_hparams)
    dev_data_generator_fn = data_reader.make_model_input_fn(tf.estimator.ModeKeys.EVAL, model_hparams)
    base_model_lib.train(train_data_generator, dev_data_generator_fn, model_hparams)


if __name__ == "__main__":
  tf.app.run()
