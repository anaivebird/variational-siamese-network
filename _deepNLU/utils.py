import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq


def variable_summaries(name,var, with_max_min=True):
  '''Tensor summaries for TensorBoard visualization'''
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    if with_max_min == True:
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


def linear(inputs, output_size, no_bias=False, regularizer=None, scope=None):
  """Define a linear connection."""
  with tf.variable_scope(scope or 'Linear'):
    input_size = inputs.get_shape()[1].value
    matrix = tf.get_variable('Matrix', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer(), regularizer=regularizer)
    bias_term = tf.get_variable('Bias', [output_size], initializer=tf.constant_initializer(0))
    output = tf.matmul(inputs, matrix)
    if not no_bias:
      output = output + bias_term
  return output


def encode_seq(input_seq, seq_len, word_embeddings, num_neurons=1000, bilstm=True, initializer=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope("word_embeddings"): # Embedding layer
    q_embed = tf.nn.embedding_lookup(params=word_embeddings, ids=input_seq, name="q_embedded") # [batch_size, pad_len, w2v_dim]

  with tf.variable_scope("encoder"): # Encoder layer

    if bilstm==True:
      cell_fw = tf.nn.rnn_cell.LSTMCell(num_neurons, initializer=initializer)
      cell_bw = tf.nn.rnn_cell.LSTMCell(num_neurons, initializer=initializer)
      (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, q_embed, sequence_length=seq_len, dtype=tf.float32)
      encoder_output = tf.concat([output_fw, output_bw], axis=-1)
      encoder_state = tf.contrib.rnn.LSTMStateTuple(tf.concat([state[0].c, state[1].c], axis=1), tf.concat([state[0].h, state[1].h], axis=1))

    else:
      cell_fw = tf.nn.rnn_cell.LSTMCell(2*num_neurons, initializer=initializer) # LSTM cell
      encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_fw, q_embed, sequence_length=seq_len, dtype=tf.float32) # [batch_size, pad_len, 2*num_neurons]

    max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()
    encoded_output = tf.contrib.layers.flatten(max_pool(encoder_output))
    encoded_state = encoder_state[0] # for VAD + CLF
    return encoded_output, encoded_state # encoder_state tuple tensor: last state, output tuple --> encoded_state [batch_size, 2*num_neurons]


def decode_seq(decoder_inputs, decoder_init_state, hidden_size, vocab_size, initializer=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope("decoder"): # Decoder layer (train)
    cell_fw2 = tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=initializer) # LSTM cell.   decoder num_neurons = hidden_size = intent dim
    decoder_output, _ = legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs, initial_state=decoder_init_state, cell=cell_fw2) # [batch_size, pad_len-1, hidden_size]
    decoder_output = tf.stack(decoder_output, axis=1) # [batch_size, pad_len-1, hidden_size]

  with tf.variable_scope("linear_projection"): # Projection layer
    W_proj =tf.get_variable("weights",[1,hidden_size, vocab_size], initializer=initializer) # hidden_size to vocab_size
    logits = tf.nn.conv1d(decoder_output, W_proj, 1, "VALID", name="logits") # project [batch_size, pad_length-1, vocab_size]

  return logits


def mlp(inputs, mlp_hidden=[], mlp_nonlinearity=tf.nn.relu, regularizer=None, scope=None):
  """Define an MLP."""
  with tf.variable_scope(scope or 'Linear'):
    mlp_layer = len(mlp_hidden)
    res = inputs
    for l in range(mlp_layer):
      res = mlp_nonlinearity(linear(res, mlp_hidden[l], scope='l'+str(l), regularizer=regularizer))
    return res
