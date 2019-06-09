# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def learning_rate_schedule(hparams):
  """Learning rate schedule based on hparams."""
  schedule_string = hparams.learning_rate_schedule
  names = schedule_string.split("*")
  names = [name.strip() for name in names if name.strip()]
  ret = tf.constant(1.0)
  for name in names:
    ret *= learning_rate_factor(name, hparams)
  return ret


def learning_rate_factor(name, hparams):
  """Compute the designated learning rate factor from hparams."""
  if name == "legacy":
    return legacy_learning_rate_schedule(hparams)
  else:
    raise ValueError("unknown learning rate factor %s" % name)


def legacy_learning_rate_schedule(hparams):
  """Backwards-compatible learning-rate schedule."""
  step_num = tf.to_float(tf.train.get_or_create_global_step())
  warmup_steps = tf.to_float(hparams.learning_rate_warmup_steps)
  ret = 5000.0 * hparams.hidden_size**-0.5 * tf.minimum(
      (step_num + 1) * warmup_steps**-1.5, (step_num + 1)**-0.5)
  optimizer_correction = 0.002 if "adam" in hparams.optimizer else 1.0
  tf.logging.info("Base learning rate: %f", hparams.learning_rate)
  return ret * optimizer_correction * hparams.learning_rate

