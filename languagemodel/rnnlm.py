import time
import tensorflow as tf

from config import *

tf.logging.set_verbosity(tf.logging.INFO)

def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context, sequence = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"image/data": tf.FixedLenFeature([], dtype=tf.string)},
        sequence_features={"image/caption_ids": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
    )
    caption = sequence["image/caption_ids"]
    image_vector = tf.decode_raw(context["image/data"], tf.float32)
    image_vector.set_shape([IMAGE_VECTOR_SIZE, ])
    return image_vector, caption


def inputs(file_pattern, num_epochs):

    if not num_epochs: num_epochs = None
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        print("Found no input files matching %s" %file_pattern)
    else:
        print("Prefetching values from %d files matching %s" %(len(data_files), file_pattern))

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(data_files, num_epochs=num_epochs)
        # Even when reading in multiple threads, share the filename queue.
        image_vector, caption = read_and_decode(filename_queue)

        caption_length = tf.shape(caption)[0]
        input_length = tf.expand_dims(tf.sub(caption_length, 1), 0)
        input_seq = tf.slice(caption, [0], input_length)
        target_seq = tf.slice(caption, [1], input_length)
        indicator = tf.ones(input_length, dtype=tf.int32)

        image_vectors, input_seqs, target_seqs, mask = tf.train.batch(
            [image_vector, input_seq, target_seq, indicator],
            batch_size=BATCH_SIZE,
            capacity=1000 + 3 * BATCH_SIZE,
            dynamic_pad=True,
            name="batch_and_pad")

        return image_vectors, input_seqs, target_seqs, mask


def MakeFancyRNNCell(hidden_units, is_train, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, forget_bias=0.0,state_is_tuple=True)
    if is_train:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    return cell


class LanguageModel(object):

    def __init__(self, mode, num_lstm_layers=1):
        assert mode=="train" or mode=="eval" or mode=="inference"
        self._mode = mode
        #initializer config
        self.initializer_scale = 0.08
        self.initializer = tf.random_uniform_initializer(
            minval=-self.initializer_scale,
            maxval=self.initializer_scale)
        # lstm config
        self.num_layers = num_lstm_layers
        self.lstm_dropout_keep_prob = 0.7
        self.clip_gradients = 5.0
        #training config
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        #with tf.name_scope("Training_Parameters"):
        #self.learning_rate_ = tf.constant(0.1, name="learning_rate")
        #self.dropout_keep_prob_ = tf.constant(0.7, name="dropout_keep_prob")

    def BuildCoreGraph(self):
        if self._mode == "inference":
            # In inference mode, a single image vector and the start token are fed in
            image_feed = tf.placeholder(dtype=tf.string, shape=[None], name="image_feed") #image vector
            input_wordid = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed")
            # Process image and insert batch dimensions.
            image_vector = tf.decode_raw(image_feed, tf.float32)
            image_vector.set_shape([1,IMAGE_VECTOR_SIZE])
            self.image_vectors_ = image_vector
            #insert batch dims as shape[0] into wordid
            self.input_seq_ = tf.expand_dims(input_wordid, 0)
            # No target sequences or input mask in inference mode.
            self.target_seqs_ = None
            self.mask = None
        else:
            self.image_vectors_, self.input_seq_, self.target_seq_, self.mask_ = \
                inputs(TRAIN_FILES if mode=="train" else VAL_FILES, num_epochs=None)

        with tf.name_scope("image_embedding"):
            # image embedding layer
            image_embedding_w_ = tf.get_variable("image_embedding_w", shape=[IMAGE_VECTOR_SIZE, HIDDEN_UNITS],
                                                 dtype=tf.float32,
                                                 initializer=tf.contrib.layers.xavier_initializer(seed=1234))
            image_embedding_b_ = tf.get_variable("image_embedding_b", shape=[HIDDEN_UNITS], dtype=tf.float32,
                                                 initializer=tf.zeros)
            self.image_embedding_ = tf.nn.relu(
                tf.matmul(self.image_vectors_, image_embedding_w_) + image_embedding_b_)

        with tf.name_scope("word_embedding"):
            self.word_embedding_ = tf.get_variable("word_embedding", shape=[VOCAB_SIZE, HIDDEN_UNITS],
                                                   dtype=tf.float32,
                                                   initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
            self.input_seq_embedding_ = tf.nn.embedding_lookup(self.word_embedding_, self.input_seq_)

        cell = MakeFancyRNNCell(HIDDEN_UNITS, self._mode == "train", self.lstm_dropout_keep_prob, self.num_layers)
        with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
            batch_size_ = self.image_vectors_.get_shape()[0]
            zero_state = cell.zero_state(batch_size_, tf.float32)

            # run the image embedding through the LSTM once to get the initial state
            _, self.initial_h_ = cell(self.image_embedding_, zero_state)

            # Allow the LSTM variables to be reused.
            lstm_scope.reuse_variables()

            sequence_length = tf.reduce_sum(self.mask_, 1)
            self.outputs_, self.final_h_ = tf.nn.dynamic_rnn(cell=cell,
                                                             inputs=self.input_seq_embedding_,
                                                             sequence_length=sequence_length,
                                                             initial_state=self.initial_h_,
                                                             scope = lstm_scope)
        with tf.variable_scope("logits"):

            self.softmax_w_ = tf.get_variable("softmax_w", shape=[HIDDEN_UNITS, VOCAB_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_b_ = tf.get_variable("softmax_b", shape=[VOCAB_SIZE], dtype=tf.float32,
                                              initializer=tf.zeros)
            # add with broadcast
            self.logits_ = matmul3d(self.outputs_, self.softmax_w_) + self.softmax_b_

            # Loss computation (true loss, for prediction)
            per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(
                self.logits_, self.target_seq_, name="per_example_loss")
            weights = tf.to_float(self.mask_)
            batch_loss = tf.div(tf.reduce_sum(tf.mul(per_example_loss_, weights)),
                                tf.reduce_sum(weights),
                                name="batch_loss")
            tf.contrib.losses.add_loss(batch_loss)
            self.total_loss_ = tf.contrib.losses.get_total_loss()
            self.target_cross_entropy_losses = per_example_loss_  # Used in evaluation.
            self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

        #finally, add a global step
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.VARIABLES])
        self.global_step_ = global_step

        print("Built model")


