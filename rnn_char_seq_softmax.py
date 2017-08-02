# Lab 12 Caharacter Sequence Softmax only

import tensorflow as tf
import numpy as np

sample = "if you want you"

idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

dic_size = len(char2idx) # RNN Input size (one hot size)
hidden_size = len(char2idx) # RNN output size
num_classes = len(char2idx) # final output size(RNN or softmax,etc)
batch_size = 1 # one sample data, one batch
sequence_length = len(sample) -1
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample] # char to index => [9, 5, 8, 7, 0, 1, 8, 6, 3, 2, 4, 8, 7, 0, 1]
x_data = [sample_idx[:-1]] # X data sample(0 ~ n-1) hello:hell
y_data = [sample_idx[1:]]  # Y label sample(1 ~ n) hello:ello

X = tf.placeholder(tf.init32,[None,sequence_length]) # x data
Y = tf.placeholder(tf.init32,[None,sequence_length]) # y label

# flatten the data(ignore batches for now). No effect if the batch size is 1
x_one_hot = tf.one_hot(X, num_classes) # one hot = 1 -> 0 1 0 0 0 0 0
X_for_softmax = tf.reshape(X_one_hot,[-1,rnn_hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w",[rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b",[num_classes])
