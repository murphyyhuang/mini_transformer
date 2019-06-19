# coding=utf-8

"""Decode from trained Tensor2Tensor style of model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from pct.utils import trainer_lib
from pct.utils import hparams_lib
from pct.utils import decoding
from pct.utils import registry
from pct import global_config

# force registration of models
from pct import models # pylint: disable=unused-import

tfe = tf.contrib.eager

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_string("config_dir", None, "Directory with training data.")

session_config = trainer_lib.create_session_config()
tf.enable_eager_execution(config=session_config)


def main(argv):
  tf.logging.set_verbosity(global_config.LOGGING_LEVEL)
  assert hasattr(FLAGS, 'config_dir'), "'config_dir' must be given to set all the hyper-parameters of this project."
  model_hparams = hparams_lib.create_hparams(FLAGS.config_dir)

  # create text2text encoders
  t2t_encoders = decoding.text2text_encoders(model_hparams)
  # create model
  model_cls = registry.model(model_hparams.model)
  model = model_cls(
    model_hparams,
    tf.estimator.ModeKeys.EVAL,
    t2t_encoders["hparams"],
  )

  # restore checkpoint
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
    checkpoint, directory=model_hparams.model_dir, max_to_keep=model_hparams.checkpoint_max_to_keep,
  )
  checkpoint.restore(checkpoint_manager.latest_checkpoint)

  decoding.decode_from_file(t2t_encoders, model, model_hparams)
