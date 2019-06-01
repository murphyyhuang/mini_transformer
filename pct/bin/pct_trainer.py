# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.utils import data_reader
from pct.utils import hparams_lib
from pct.utils import base_model_lib
from pct.utils import trainer_lib

# force registration of models
from pct import models # pylint: disable=unused-import

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_string("config_dir", None, "Directory with training data.")

session_config = trainer_lib.create_session_config()
tf.enable_eager_execution(config=session_config)


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # read hyper-parameters
  assert hasattr(FLAGS, 'config_dir'), "'config_dir' must be given to set all the hyper-parameters of this project."
  model_hparams = hparams_lib.create_hparams(FLAGS.config_dir)
  train_data_generator = data_reader.TextDataGenerator(tf.estimator.ModeKeys.TRAIN, model_hparams)

  base_model_lib.train(train_data_generator, model_hparams)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
