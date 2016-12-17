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

    images_to_captions = defaultdict(lambda: [])
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        reader = tf.TFRecordReader()
        data_files = []
        for pattern in file_pattern.split(","):
            data_files.extend(tf.gfile.Glob(pattern))
        if not data_files:
            print("Found no input files matching %s" % file_pattern)
        else:
            print("Prefetching values from %d files matching %s" % (len(data_files), file_pattern))

        filename_queue = tf.train.string_input_producer(data_files, num_epochs=1, shuffle=False)

        _, serialized_example = reader.read(filename_queue)
        context, sequence = tf.parse_single_sequence_example(
                                serialized_example,
                                context_features={"image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
                                sequence_features={"image/caption": tf.FixedLenSequenceFeature([], dtype=tf.string)}
                            )
        caption = sequence["image/caption"]
        image_id = context["image/image_id"]

        captions, image_ids = tf.train.batch(
            [caption, image_id],
            batch_size=BATCH_SIZE,
            capacity=1000 + 3 * BATCH_SIZE,
            dynamic_pad=True,
            allow_smaller_final_batch=True
            )

        # Initialize the variables (like the epoch counter).
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            totalcount = 0
            while not coord.should_stop():
                # Run training steps or whatever
                captions_val, imageid_val = sess.run([captions, image_ids])
                for cap, id in zip(captions_val, imageid_val):
                    cap = [word for word in cap if word != '']
                    images_to_captions[id].append(" ".join(cap))
                    totalcount += 1
                    if totalcount % 100 == 0:
                        print("Processed %d captions" % totalcount)
                    #print(images_to_captions[id])
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            print("writing data to file")
            print("Number of unique image filenames: %d" % len(images_to_captions.keys()))
            with open(output_file, 'w') as f:
                json.dump(images_to_captions, f)

            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)


def DoInference(input_file, output_file):
    with open(input_file, 'r') as f:
        ground_truth = json.load(f)

    predictions = {}
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

        total_bleu_score = 0.0
        total_count = 0
        for imageid, ref_captions in ground_truth.items():
            imagefile = "COCO_val2014_%012d.jpg" % int(imageid)
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
            bleu_score = nltk.translate.bleu_score.sentence_bleu([x.split(" ") for x in ref_captions], predicted_caption)
            #print([x.split(" ") for x in ref_captions])
            #print(predicted_caption)

            predictions[imagefile] = {'reference caption': ref_captions,
                                        'predicted_caption': " ".join(predicted_caption),
                                        'bleu_score': bleu_score}
            #print(predictions[imagefile])
            total_bleu_score += bleu_score
            total_count += 1

            if total_count % 100 == 0:
                print("Processed %d items" %total_count)

        print("AVERAGE BLEU score: %f" %(total_bleu_score/total_count))
        with open(output_file, 'w') as f:
             json.dump(predictions, f)



def main(unused_argv):
    if not os.path.isfile(PROCESSED_TEST_FILE):
        print("extracting image filenames and captions from %s into %s" %(TEST_FILES,PROCESSED_TEST_FILE))
        ExtractData(TEST_FILES, PROCESSED_TEST_FILE)

    DoInference(PROCESSED_TEST_FILE, PREDICTIONS_FILE)

if __name__ == "__main__":
    tf.app.run()


'''
Bad way to do this - teeribly inefficeint
def ExtractData(file_pattern, output_file):
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))

    totalcount = 0
    images_to_captions = defaultdict(lambda: [])
    with tf.Session() as sess:
        for file in data_files[:1]:
            print("processing file %s" %file)
            for record in tf.python_io.tf_record_iterator(file):
                context, sequence = tf.parse_single_sequence_example(
                    record,
                    context_features={"image/data": tf.FixedLenFeature([], dtype=tf.string),
                                      "image/image_id": tf.FixedLenFeature([], dtype=tf.int64)},
                    sequence_features={"image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                       "image/caption": tf.FixedLenSequenceFeature([],dtype = tf.string)}
                )
                caption = sequence["image/caption"]
                image_id = context["image/image_id"]
                caption = sess.run(caption)
                id = sess.run(image_id)
                image_filename = "COCO_val2014_%012d.jpg" %id
                images_to_captions[image_filename].append(list(caption))
                totalcount += 1
                if totalcount %100 == 0:
                    print("Processed %d captions" %totalcount)
        print("writing data to file")
        print("Number of unique image filenames: %d" %len(images_to_captions.keys()))
        with open(output_file, 'w') as f:
            json.dump(images_to_captions, f)
'''
