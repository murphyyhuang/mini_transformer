# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pct.utils import data_reader
import tensorflow as tf


class Hparams_data_reader():

  def __init__(self):
    self.num_threads = 4
    self.problem = 'translate_enzh_wmt32k'
    self.data_dir = '/home/murphyhuang/dev/mldata/en_ch_translate'
    self.shuffle_files = True
    self.shuffle_buffer_size = 32


def create_generator_test():
  hparams = Hparams_data_reader()
  dataset = data_reader.create_generator(hparams, 'train')
  iterator = dataset.make_one_shot_iterator()

  sess = tf.Session()
  # sess.run(iterator.initializer)
  a = sess.run(iterator.get_next())
  print(a)


if __name__ == '__main__':
  create_generator_test()
