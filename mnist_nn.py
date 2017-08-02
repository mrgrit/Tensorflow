import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10
learning_rate = 0.01
training_epochs = 10
batch_size = 100

X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32,[None,nb_classes])

# deep & wide NN
W1 = tf.Variable(tf.random_normal([784,256]))
b1 = tf.Variable(tf.random_normal([256]))
# relu
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([256,256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)

W3 = tf.Variable(tf.random_normal([256,10]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch) :
            # batch 크기만큼 데이터셋을 읽어옴
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost,optimizer],feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c/total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost= ','{:.9f}'.format(avg_cost))
    print("learning finished")
    # Test the model using thest sets
    print("Accuracy: ", accuracy.eval(session = sess, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}))
