# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from pct.utils import registry
from pct.utils import hparams_lib


def train(train_generator,
          dev_data_generator_fn,
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
  # checkpoint_prefix = os.path.join(hparams.model_dir, 'ckpt')
  step_counter = tf.train.get_or_create_global_step()
  checkpoint = tf.train.Checkpoint(
    model=model, optimizer=optimizer, optimizer_step=step_counter
  )
  checkpoint_manager = tf.contrib.checkpoint.CheckpointManager(
    checkpoint, directory=hparams.model_dir, max_to_keep=hparams.checkpoint_max_to_keep,
  )
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(checkpoint_manager.latest_checkpoint)

  start_step = step_counter.value()
  for step_index in range(start_step, hparams.train_steps):
    features = train_generator.get_next()

    with tf.GradientTape() as tape:
      logits, loss = model(features, training=True)
      tf.contrib.summary.scalar("loss", loss)
      print("* Loss for step {}: {}".format(step_index, loss))
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(
        zip(grads, model.trainable_variables),
        global_step=step_counter)

    if not step_index % hparams.eval_steps:
      dev_data_generator = dev_data_generator_fn()

      eval_losses = []

      while True:
        try:
          eval_features = dev_data_generator.get_next()
          _, eval_loss = model(eval_features, training=False)
          eval_losses.append(eval_loss)
        except tf.errors.OutOfRangeError:
          break

      print("* Eval for step {}, loss: {}".format(step_index, tf.reduce_mean(eval_losses)))
      checkpoint_manager.save()
