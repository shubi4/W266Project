# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""A binary to evaluate Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from inception import inception_eval
from inception.flowers_data import FlowersData

#import inception_eval
#from flowers_data import FlowersData


tf.app.flags.DEFINE_string('data_dir', '/w266/project/flowers_data/',
                           'Output data directory')
tf.app.flags.DEFINE_string('subset', 'validation', 'train or validation')
tf.app.flags.DEFINE_string('eval_dir', '/w266/project/flowers_data/eval/',
                           'Eval directory')
tf.app.flags.DEFINE_integer('num_examples', 100,
                           'num examples')
tf.app.flags.DEFINE_string('checkpoint_dir', '/w266/project/inception-v3-model/inception-v3',
                           'checkpoint directory')
tf.app.flags.DEFINE_integer('input_queue_memory_factor', 1, '')
tf.app.flags.DEFINE_integer('run_once', True, '')


FLAGS = tf.app.flags.FLAGS

def main(unused_argv=None):
  dataset = FlowersData(subset=FLAGS.subset)
  assert dataset.data_files()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  inception_eval.evaluate(dataset)


if __name__ == '__main__':
  tf.app.run()
