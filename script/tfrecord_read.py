# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def main_1():
  data_dir = "/home/murphyhuang/dev/mldata/en_ch_translate/translate_enzh_wmt32k-train-00000-of-00100"
  dataset = tf.data.TFRecordDataset(tf.constant(data_dir))
  iterator = dataset.make_initializable_iterator()
  sess = tf.Session()
  sess.run(iterator.initializer)
  print(sess.run(iterator.get_next()))


def main_2():
  data_dir = "/home/murphyhuang/dev/mldata/en_ch_translate/translate_enzh_wmt32k-train-00000-of-00100"
  record_iterator = tf.python_io.tf_record_iterator(path=data_dir)

  for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)

    print(example)

if __name__ == '__main__':
  main_2()
