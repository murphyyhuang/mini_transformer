# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pct.utils import data_reader
from pct.utils import hparam
import tensorflow as tf

tf.enable_eager_execution()


def create_generator_test():
  yaml_dir = os.path.join(os.path.dirname(__file__), '../test_data/hparams.yml')
  hparams = hparam.HParams()
  hparams.from_yaml(yaml_dir)

  data_generator = data_reader.TextDataGenerator('train', hparams)

  index = 1
  sentence_counter = []
  while True:
    try:
      result = data_generator.get_next()
      print('* Round: {}'.format(index))
      print('* Input shape: {}'.format(result[0]['inputs'].shape))
      print('* Target shape: {}'.format(result[0]['targets'].shape))
      sentence_counter.append(result[0]['inputs'].shape[0])
      index += 1
    except:
      break

  print('* The total number of sentences: {}'.format(sum(sentence_counter)))


if __name__ == '__main__':
  create_generator_test()
