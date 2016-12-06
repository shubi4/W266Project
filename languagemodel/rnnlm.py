import time

import tensorflow as tf
import numpy as np

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

def MakeFancyRNNCell(H, keep_prob, num_layers=1):
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
  #### YOUR CODE HERE ####
  cell = None  # replace with something better

  # Solution
  cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0,state_is_tuple=True)
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob,
                                       output_keep_prob=keep_prob)
  cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

  #### END(YOUR CODE) ####
  return cell


def affine_layer(hidden_dim, x, seed=0):
    # x: a [batch_size x # features] shaped tensor.
    # hidden_dim: a scalar representing the # of nodes.
    # seed: use this seed for xavier initialization.

    # START YOUR CODE
    # print "hidden_dim: %s" %type(hidden_dim)
    num_features = x.get_shape()[-1]
    # print "num features: %s", type(num_features)

    w_var = tf.get_variable("w", shape=[num_features, hidden_dim],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer(seed=seed))
    # print "w_var: %s" %type(w_var)
    b_var = tf.get_variable("b", shape=[1, hidden_dim], dtype=tf.float32,
                            initializer=tf.zeros)
    # print "b_var: %s" %type(b_var)

    z_t = tf.matmul(x, w_var) + b_var
    # print "z_t: %s" %type(z_t)

    return z_t
    # END YOUR CODE


def fully_connected_layers(hidden_dims, x):
    # hidden_dims: A list of the width of the hidden layer.
    # x: the initial input with arbitrary dimension.
    # To get the tests to pass, you must use relu(.) as your element-wise nonlinearity.
    #
    # Hint: see tf.variable_scope - you'll want to use this to make each layer
    # unique.
    # Hint: a fully connected layer is a nonlinearity of an affine of its input.
    #       your answer here only be a couple of lines long (mine is 4).

    # START YOUR CODE
    # initially, input and output are not in any scope
    input = output = x
    for i, dim in enumerate(hidden_dims):
        # create a new variable scope for this layer of the NN
        with tf.variable_scope("layer_" + str(i)):
            # Each layer is an affine tranformation followed by a non-linearity.
            # if this is the first layer, input = x (the "real" inputs)
            # For subsequent layers, input = output of the previous layer,
            # and input is in the variable scope of the *previous* layer
            # print "Before relu, input=" + input.name
            output = tf.nn.relu(affine_layer(dim, input))
            # the output from this affine layer becomes input for the next one
            # Now input's variable scope is *this* layer
            input = output
            # print "After relu, input=" + input.name
    return output
    # END YOUR CODE


class RNNLM(object):

  def __init__(self, V, H, img_embedding_size, num_layers=1):
    """Init function.

    This function just stores hyperparameters. You'll do all the real graph
    construction in the Build*Graph() functions below.

    Args:
      V: vocabulary size
      H: hidden state dimension
      num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
    """
    # Model structure; these need to be fixed for a given model.
    self.V = V
    self.H = H
    self.img_embedding_size = img_embedding_size
    self.num_layers = num_layers

    # Training hyperparameters; these can be changed with feed_dict,
    # and you may want to do so during training.
    with tf.name_scope("Training_Parameters"):
      self.learning_rate_ = tf.constant(0.1, name="learning_rate")
      self.dropout_keep_prob_ = tf.constant(0.5, name="dropout_keep_prob")
      # For gradient clipping, if you use it.
      # Due to a bug in TensorFlow, this needs to be an ordinary python
      # constant instead of a tf.constant.
      self.max_grad_norm_ = 5.0


  def BuildCoreGraph(self):
    """Construct the core RNNLM graph, needed for any use of the model.
    """

    self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

    self.img_embedding_ = tf.placeholder(tf.float32, [None, self.img_embedding_size])

    # Should be the same shape as inputs_w_
    self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

    # Get dynamic shape info from inputs
    with tf.name_scope("batch_size"):
      self.batch_size_ = tf.shape(self.input_w_)[0]
    with tf.name_scope("max_time"):
      self.max_time_ = tf.shape(self.input_w_)[1]

    # Get sequence length from input_w_.
    # This will be a vector with elements ns[i] = len(input_w_[i])
    # You can override this in feed_dict if you want to have different-length
    # sequences in the same batch, although you shouldn't need to for this
    # assignment.
    self.ns_ = tf.tile([self.max_time_], [self.batch_size_,], name="ns")

    #### YOUR CODE HERE ####
    # Construct embedding layer
    self.embedding_ = tf.get_variable("embedding", shape=[self.V, self.H], 
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    self.input_x_ = tf.nn.embedding_lookup(self.embedding_, self.input_w_)
    
    # Construct RNN/LSTM cell and recurrent layer (hint: use tf.nn.dynamic_rnn)
    # code
    self.img_embedding_out_ = fully_connected_layers([self.H], self.img_embedding_)

    cell = MakeFancyRNNCell(self.H, self.dropout_keep_prob_, self.num_layers)
    zero_state = cell.zero_state(self.batch_size_, tf.float32)
    #run the image embedding through the LSTM once to get the initial state
    _, self.initial_h_ = cell(self.img_embedding_out_, zero_state)

    #print self.initial_h_

    print cell.state_size
    self.outputs_, self.final_h_ = tf.nn.dynamic_rnn(cell= cell, inputs= self.input_x_,
                                sequence_length= self.ns_, initial_state = self.initial_h_)
    #print type(self.final_h_)
    #print self.final_h_

    # Softmax output layer, over vocabulary
    # Hint: use the matmul3d() helper here.
    self.softmax_w_ = tf.get_variable("softmax_w", shape=[self.H, self.V],
                            dtype=tf.float32, 
                            initializer=tf.contrib.layers.xavier_initializer())
    self.softmax_b_ = tf.get_variable("softmax_b", shape=[self.V], dtype=tf.float32,
                            initializer=tf.zeros)

    # add with broadcast
    self.logits_ = matmul3d(self.outputs_, self.softmax_w_) + self.softmax_b_
    
    # Loss computation (true loss, for prediction)
    per_example_loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits_,
                    self.target_y_,name="per_example_loss")
    self.loss_ = tf.reduce_sum(per_example_loss_, name="loss")
    
    #### END(YOUR CODE) ####


  def BuildTrainGraph(self):
    """Construct the training ops.

    You should define:
    - train_loss_ (optional): an approximate loss function for training
    - train_step_ : a training op that can be called once per batch

    Your loss function should return a *scalar* value that represents the
    _summed_ loss across all examples in the batch (i.e. use tf.reduce_sum, not
    tf.reduce_mean).
    """
    # Replace this with an actual training op
    self.train_step_ = tf.no_op(name="dummy")

    # Replace this with an actual loss function
    self.train_loss_ = None

    #### YOUR CODE HERE ####

    # Define loss function(s)
    with tf.name_scope("Train_Loss"):
      # Placeholder: replace with a sampled loss
      #self.train_loss = self.loss_
      per_example_train_loss_ = tf.nn.sampled_softmax_loss(
                                    weights= tf.transpose(self.softmax_w_), 
                                    biases= self.softmax_b_, 
                                    inputs= tf.reshape(self.outputs_, [-1,self.H]),
                                    labels= tf.reshape(self.target_y_,[-1,1]),
                                    num_sampled= 100,
                                    num_classes= self.V,
                                    num_true= 1,
                                    name="per_example_train_loss")
    self.train_loss_ = tf.reduce_sum(per_example_train_loss_, name="train_loss")


    # Define optimizer and training op
    with tf.name_scope("Training"):
        #self.train_step_ = None  # Placeholder: replace with an actual op
        self.optimizer_ = tf.train.AdagradOptimizer(self.learning_rate_)
        self.train_step_ = self.optimizer_.minimize(self.train_loss_)
    #### END(YOUR CODE) ####


  def BuildSamplerGraph(self):
    """Construct the sampling ops.

    You should define pred_samples_ to be a Tensor of integer indices for
    sampled predictions for each batch element, at each timestep.

    Hint: use tf.multinomial, along with a couple of calls to tf.reshape
    """
    # Replace with a Tensor of shape [batch_size, max_time, 1]
    self.pred_samples_ = None

    #### YOUR CODE HERE ####
    
    self.pred_samples_ = tf.multinomial(tf.reshape(self.logits_, [-1, self.V]), num_samples=1)
    self.pred_samples_ = tf.reshape(self.pred_samples_, [self.batch_size_, self.max_time_,1])
    
    
    #### END(YOUR CODE) ####

