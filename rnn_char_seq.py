import tensorflow as tf
import numpy as np

sample = "if you want you"
idx2char = list(set(sample)) # set으로 만들어 다시 list로 만듬 => ['o', 'u', 'n', 'a', 't', 'f', 'w', 'y', ' ', 'i']
char2idx = {c:i for i, c in enumerate(idx2char)} # dict로 만듬 => {'o': 0, 'u': 1, 'n': 2, 'a': 3, 't': 4, 'f': 5, 'w': 6, 'y': 7, ' ': 8, 'i': 9}

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

x_hone_hot = tf.one_hot(X,num_classes) # one hot : 1 -> 0 1 0 0 0 0 0 0
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size,tf.float32)

# FC layer
X_for_fc = tf.reshape(outputs,[-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(outputs, num_classes, activation_fn=None)

output = tf.reshape(outputs,[batch_size,sequence_length,num_classes])

weight = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, target=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(lenarning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs,axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l,_ = sess.run([loss,train],feed_dict = {X:x_data, Y:y_data})
        result = sess.run(prediction,feed_dict = {X:x_data})

        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss: ", l, "Prediction: ", ''.join(result_str))
