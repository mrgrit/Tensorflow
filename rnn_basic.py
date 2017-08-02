import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint
pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

with tf.variable_scope('one_cell') as scope:
    # One cell RNN input_dim (4) -> output_dim(2)
    hidden_size = 2 # output size [X,X]
    # Cell 생성
    cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)
    print(cell.output_size, cell.state_size)
    # h = [1, 0, 0, 0]
    x_data = np.array([[h]],dtype=np.float32) # x_data = [[[1,0,0,0]]]
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

with tf.variable_scope('two_sequances') as scope:
    # One Cell RNN input_dim (4) ->  output_dim(2).  sequence : 5
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units = hidden_size)
    x_data = np.array([[h,e,l,l,o]],dtype=np.float32) # sequence_length=5
    print(x_data.shape)
    pp.pprint(x_data)
    outputs, states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())

with tf.variable_scope('3_batches') as scope:
    # one Cell RNN input_dim(4) -> output_dim(2). sequence:5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([
    [h,e,l,l,o],
    [e,o,l,l,l],
    [l,l,e,e,l]],dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units = hidden_size,state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())


with tf.variable_scope('3_batches_dynamic_length') as scope:
    # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch 3
    # 3 batches 'hello', 'eolll', 'lleel'
    x_data = np.array([[h, e, l, l, o],
                       [e, o, l, l, l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)

    hidden_size = 2
    cell = rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)
    outputs, _states = tf.nn.dynamic_rnn(cell,x_data,sequence_length[5,3,4],dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    pp.pprint(outputs.eval())
