import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
import datetime
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

nb_classes = 10
learning_rate = 0.001
training_epochs = 50
batch_size = 200

X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32,[None,nb_classes])

# dropout (keep_prob) rate 0.7 on training, but should be 1 for testing
# 학습 / Test 에 서로다른 dropout rate를 feed_dict로 주기위해 placeholder로 설정
keep_prob = tf.placeholder(tf.float32)

# deep & wide NN
# Xavier for initialize weight
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
# relu
L1 = tf.nn.relu(tf.matmul(X,W1)+b1)
# drop out layer
L1 = tf.nn.dropout(L1, keep_prob = keep_prob)

W2 = tf.get_variable("W2", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.relu(tf.matmul(L1,W2)+b2)
# drop out layer
L2 = tf.nn.dropout(L2, keep_prob = keep_prob)

W3 = tf.get_variable("W3", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.relu(tf.matmul(L2,W3)+b3)
# drop out layer
L3 = tf.nn.dropout(L3, keep_prob = keep_prob)

W4 = tf.get_variable("W4", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.relu(tf.matmul(L3,W4)+b4)
# drop out layer
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

W5 = tf.get_variable("W5", shape=[512,512], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([512]))
L5 = tf.nn.relu(tf.matmul(L4,W5)+b5)
# drop out layer
L5 = tf.nn.dropout(L5, keep_prob = keep_prob)

W6 = tf.get_variable("W6", shape=[512,10], initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L5, W6) + b6

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
            # feed_dict에 keep_prob 가 추가됨
            feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.7}
            c,_ = sess.run([cost,optimizer],feed_dict=feed_dict)
            avg_cost += c/total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost= ','{:.9f}'.format(avg_cost),'now : ',datetime.datetime.now())
    print("learning finished")
    # Test the model using thest sets
    # Test 때는 keep prob를 1로 준다.
    print("Accuracy: ", accuracy.eval(session = sess, feed_dict = {X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

"""
    # Get one and prodict
    r = random.randint(0, mnist.test.num_examples -1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {X:mnist.test.image[r:r+1], keep_prob : 1}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()
"""
