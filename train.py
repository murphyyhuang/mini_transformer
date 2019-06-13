# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf

from pct.bin import pct_trainer


def main(argv):
  pct_trainer.main(argv)


if __name__ == "__main__":

  config_path = os.path.join(
    os.path.dirname(__file__),
    'pct/test_data/libigpu0_hparams.yml'
  )
  sys_argv = """
  --config_dir=/home/murphyhuang/dev/src/github.com/EstelleHuang666/PCT-dev/pct/test_data/hparams.yml
  """
  sys_argv = sys_argv.split()
  sys.argv.extend(sys_argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
