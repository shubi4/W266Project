from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
import sys
import numpy as np
import tensorflow as tf
import json
import nltk

from config import *
from rnnlm import LanguageModel
from vocabulary import Vocabulary
from collections import defaultdict

tf.logging.set_verbosity(tf.logging.INFO)

def ExtractData(file_pattern, output_file):
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))

    images_to_captions = defaultdict(lambda: [])
    with tf.Session() as sess:
        for file in data_files[:1]:
            print("processing file %s" %file)
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
                image_id = context["image/image_id"]
                caption = sess.run(caption)
                id = sess.run(image_id)
                image_filename = "COCO_val2014_%012d.jpg" %id
                images_to_captions[image_filename].append(list(caption))
                print(images_to_captions[image_filename])
        print("writing data to file")
        with open(output_file, 'w') as f:
            json.dump(images_to_captions, f)


def DoInference(input_file, output_file):
    with open(input_file, 'r') as f:
        ground_truth = json.load(f)

    predictions = defaultdict(lambda: [])
    vocab = Vocabulary(VOCAB_FILE)

    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        model_path = CHECKPOINT_FILE
        # Build the model for evaluation.
        model = LanguageModel(mode="inference")
        model.BuildCoreGraph()
        model.BuildSamplerGraph()

        # Create the Saver to restore model Variables.
        saver = tf.train.Saver()

        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step_.name)
        tf.logging.info("Successfully loaded %s at global step = %d.", os.path.basename(model_path), global_step)

        for imagefile, captions in ground_truth:
            vector_filename = os.path.join(INFERENCE_VECTOR_FILES_DIR, imagefile)
            image_vector = np.fromfile(vector_filename, dtype=np.float32)
            #print(image_vector.shape)
            assert image_vector.shape == (2048,)

            captionids = []
            h = sess.run(model.initial_h_, feed_dict = {model.image_feed_: image_vector})
            wordid = np.array([vocab.start_id])

            captionids.append(wordid[0])
            for i in xrange(50):
                feed = {model.initial_h_: h, model.input_wordid_: wordid}
                wordid, h = sess.run([model.pred_samples_, model.final_h_], feed_dict=feed)
                captionids.append(wordid[0])
                if wordid[0] == vocab.end_id:
                    break

            predicted_caption = [vocab.id_to_word(word_id) for word_id in captionids]
            predictions[imagefile].append(" ".join(predicted_caption))

            for ref in ground_truth[imagefile]:
                ref = ref.split(" ")
            # there may be several references
            #BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
            #print
            #BLEUscore
            #print(caption)
            print("Predicted caption: "),
            print(" ".join(vocab.id_to_word(word_id) for word_id in caption))
            print()
            print()

def main(unused_argv):
    if not os.path.isfile(PROCESSED_TEST_FILE):
        print("extracting image filenames and captions from %s into %s" %(TEST_FILES,PROCESSED_TEST_FILE))
        ExtractData(TEST_FILES, PROCESSED_TEST_FILE)

    DoInference(PROCESSED_TEST_FILE, PREDICTIONS_FILE)

if __name__ == "__main__":
    tf.app.run()