# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pct.utils import registry
from pct.utils import hparams_lib


def train(train_generator,
          hparams,
          ):
  hparams = hparams_lib.copy_hparams(hparams)
  generator_hparams = train_generator.get_generator_hparams()
  model_cls = registry.model(hparams.model)
  model = model_cls(
    hparams,
    tf.estimator.ModeKeys.TRAIN,
    generator_hparams,
  )

  optimizer = model.optimizer()
  # Create and restore checkpoint (if one exists on the path)
  checkpoint_prefix = os.path.join(hparams.model_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
    model=model, optimizer=optimizer, step_counter=step_counter
  )
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(hparams.model_dir))
  for step_index in range(hparams.train_steps):
    features = train_generator.get_next()

    with tf.GradientTape() as tape:
      logits, losses = model(features)
      grads = tape.gradient(losses, model.trainable_variables)
      optimizer.apply_gradients(
        zip(grads, model.trainable_variables),
        global_step=step_counter)

    if not step_index % hparams.eval_steps:
      checkpoint.save(checkpoint_prefix)
