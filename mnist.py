import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 읽어올때 부터 one hot 처리를 해서 읽어오겠다.
# 0~9
nb_classes = 10

# 784 => 28*28 픽셀
X = tf.placeholder(tf.float32,[None, 784])
Y = tf.placeholder(tf.float32,[None,nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y,1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 30
# batch_size를 줄이면 속도는 느려지나 cost와 accuracy가 좋아짐
batch_size = 100 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Training
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            #batch 크기만큼 데이터셋을 읽어옴
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c,_ = sess.run([cost,optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c/total_batch
        print('Epoch: ', '%04d' % (epoch + 1), 'cost= ','{:.9f}'.format(avg_cost))
    print("learning finished")
    # Test the model using thest sets
    print("Accuracy:", accuracy.eval(session = sess, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}))

      
    # Test 시각화
    r = random.randint(0, mnist.test.num_examples -1)
    print("Lable1:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis,1), feed_dict = {X:mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap = 'Greys', interpolation = 'nearest')
    plt.show()

