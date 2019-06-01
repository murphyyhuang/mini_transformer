# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2


def create_session_config(log_device_placement=True,
                          enable_graph_rewriter=False,
                          gpu_mem_fraction=0.95,
                          xla_jit_level=tf.OptimizerOptions.OFF,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0):
  if enable_graph_rewriter:
    rewrite_options = rewriter_config_pb2.RewriterConfig()
    rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.ON
    graph_options = tf.GraphOptions(rewrite_options=rewrite_options)
  else:
    graph_options = tf.GraphOptions(
      optimizer_options=tf.OptimizerOptions(
        opt_level=tf.OptimizerOptions.L1,
        do_function_inlining=False,
        global_jit_level=xla_jit_level,
      )
    )

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_fraction)

  config = tf.ConfigProto(
    allow_soft_placement=True,
    graph_options=graph_options,
    gpu_options=gpu_options,
    log_device_placement=log_device_placement,
    inter_op_parallelism_threads=inter_op_parallelism_threads,
    intra_op_parallelism_threads=intra_op_parallelism_threads,
    isolate_session_state=True
  )

  return config

