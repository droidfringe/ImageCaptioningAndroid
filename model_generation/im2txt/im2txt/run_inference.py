# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

import cv2
import numpy as np

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    #output_node_names = [n.name for n in g.as_graph_def().node]


    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      #with tf.gfile.GFile(filename, "rb") as f:
      #  image = f.read()
      image = cv2.imread(filename)
      image = image.astype(np.float32)
      image = image / 255.0
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


    """
    input_names = {'image_feed:0': (), 'input_feed:0': (1,), 'lstm/state_feed:0': (1,1024)}
    input_tensors = {x: g.get_tensor_by_name(x) for x in input_names}
    for n,t in input_tensors.items():
      print('old: ', t, tf.shape(t))
      tf.reshape(t, input_names[n])
      print('new: ', t, tf.shape(t))

    g.finalize()
    """
    #output_tensors = [g.get_tensor_by_name(x) for x in output_names]

    output_node_names = ["lstm/initial_state", "softmax", "lstm/state"]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
      sess,
      sess.graph_def,
      output_node_names)

    # Save the frozen graph
    with open('/home/droid/show_and_tell/im2txt/im2txt/data/output_graph_4.pb', 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())


if __name__ == "__main__":
  tf.app.run()
