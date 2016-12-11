
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from config import *
from rnnlm import LanguageModel


def main(unused_argv):

    # Create training directory.
    if not tf.gfile.IsDirectory(CHECKPOINT_DIR):
        print("Creating checkpoint directory: %s" %CHECKPOINT_DIR)
        tf.gfile.MakeDirs(CHECKPOINT_DIR)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = LanguageModel()
        model.BuildCoreGraph(TRAIN_FILES)

        # Set up the learning rate.
        learning_rate_decay_fn = None
        learning_rate = tf.constant(model.initial_learning_rate)
        if model.learning_rate_decay_factor > 0:
            num_batches_per_epoch = NUM_TRAIN_EXAMPLES / BATCH_SIZE
            decay_steps = int(num_batches_per_epoch * model.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=model.learning_rate_decay_factor,
                    staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss_,
            global_step=model.global_step_,
            learning_rate=learning_rate,
            optimizer="SGD",
            clip_gradients=model.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=MAX_CHECKPOINTS_TO_KEEP)

        print("Starting training")
        # Run training.
        tf.contrib.slim.learning.train(
            train_op,
            CHECKPOINT_DIR,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            graph=g,
            global_step=model.global_step_,
            number_of_steps=NUMBER_OF_STEPS,
            saver=saver)


if __name__ == "__main__":
    tf.app.run()