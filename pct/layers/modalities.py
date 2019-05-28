# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pct.layers import common_attention
from pct.layers import common_layers


def generic_loss(top_out, targets, model_hparams, vocab_size, weights_fn):
  """Compute loss numerator and denominator for one shard of output."""
  del vocab_size  # unused arg
  logits = top_out
  logits = common_attention.maybe_upcast(logits, hparams=model_hparams)
  # cutoff = getattr(model_hparams, "video_modality_loss_cutoff", 0.0)
  return common_layers.padded_cross_entropy(
      logits,
      targets,
      model_hparams.label_smoothing,
      cutoff=0,
      weights_fn=weights_fn)
