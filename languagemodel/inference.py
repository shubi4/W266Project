from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import sys
import numpy as np
import tensorflow as tf

from config import *
from rnnlm import LanguageModel
from vocabulary import Vocabulary


tf.logging.set_verbosity(tf.logging.INFO)


def inference_on_dataset(file_pattern):
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))

    with tf.Session() as sess:
        for file in data_files:
            for i, record in enumerate(tf.python_io.tf_record_iterator(file)):
                context, sequence = tf.parse_single_sequence_example(
                    record,
                    context_features={"image/data": tf.FixedLenFeature([], dtype=tf.string),
                                      "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
                    sequence_features={"image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                       "image/caption": tf.FixedLenSequenceFeature([],dtype = tf.string)}
                )
                caption_ids = sequence["image/caption_ids"]
                caption = sequence["image/caption"]
                image_vector = tf.decode_raw(context["image/data"], tf.float32)
                image_vector.set_shape([IMAGE_VECTOR_SIZE, ])
                image_id = context["image/image_id"]

                #print("Caption Ids: ", sess.run(caption_ids))
                print("Caption: ", sess.run(caption))
                #print("Image vector (%d): " %len(sess.run(image_vector)))
                id = sess.run(image_id)
                image_filename = "COCO_val2014_%012d.jpg" %id
                print("Image Filename: ", image_filename)
                InferenceOnSingleFile(image_filename)
                if i > 20:
                    return


def InferenceOnSingleFile(filename):

    #assert len(sys.argv) >= 2
    #vector_filename = os.path.join(INFERENCE_VECTOR_FILES_DIR, sys.argv[1])
    vector_filename = os.path.join(INFERENCE_VECTOR_FILES_DIR, filename)
    #image_filename = os.path.join(INFERENCE_IMAGE_FILES_DIR, sys.argv[1])
    image_vector = np.fromfile(vector_filename, dtype=np.float32)
    print(image_vector.shape)
    assert image_vector.shape == (2048,)

    vocab = Vocabulary(VOCAB_FILE)

    model_path = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if not model_path:
        tf.logging.info("Skipping evaluation. No checkpoint found in: %s", CHECKPOINT_DIR)
        return

    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model = LanguageModel(mode="inference")
        model.BuildCoreGraph()
        model.BuildSamplerGraph()

        # Create the Saver to restore model Variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Load model from checkpoint.
            tf.logging.info("Loading model from checkpoint: %s", model_path)
            saver.restore(sess, model_path)
            global_step = tf.train.global_step(sess, model.global_step_.name)
            tf.logging.info("Successfully loaded %s at global step = %d.", os.path.basename(model_path), global_step)

            caption = []
            h = sess.run(model.initial_h_, feed_dict = {model.image_feed_: image_vector})
            wordid = np.array([vocab.start_id])

            caption.append(wordid[0])
            for i in xrange(50):
                feed = {model.initial_h_: h, model.input_wordid_: wordid}
                wordid, h = sess.run([model.pred_samples_, model.final_h_], feed_dict=feed)
                caption.append(wordid[0])
                if wordid[0] == vocab.end_id:
                    break

            #print(caption)
            print("Predicted caption: "),
            print(" ".join(vocab.id_to_word(word_id) for word_id in caption))
            print()
            print()

def main(unused_argv):
    inference_on_dataset(TEST_FILES)


if __name__ == "__main__":
    tf.app.run()