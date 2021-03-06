import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input placeholders
X = tf.placeholder(tf.float32,[None,784])
# -1 => n개, 28 X 28, 1 color image(black & white)
X_img = tf.reshape(X,[-1,28,28,1])
# 1 ~ 10
Y = tf.placeholder(tf.float32,[None,10])

# L1 Image shape = (?, 28, 28, 1)
# filter 3 X 3, 1color, 32EA
W1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
# Conv -> (?, 28, 28, 32) 1 X 1 / SAME
L1 = tf.nn.conv2d(X_img,W1,strides = [1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
# kenel = filter
# Pool -> (?, 14, 14, 32) - 2 X 2 / SAME
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
#Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
#Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)

# L2 ImgIn shape = (? , 14, 14, 32)
# L1 maxpool은 L2에게 이미지 이므로 32는 같은 값으로 사용
# L1 maxpool의 32는 갯수, L2 filter의 32는 색깔수(??)
W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
L2_flat = tf.reshape(L2,[-1,7*7*64])
# Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
# Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
# Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)

# Final FC 7 X 7 X 64 inputs --> 10 outputs
W3 = tf.get_variable("W3", shape=[7*7*64,10], initializer = tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat,W3)+b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Learning Started. it takes much time')
for epoch in range(training_epochs) :
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #print(type(batch_xs)) #<class 'numpy.ndarray'>
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost,optimizer], feed_dict = feed_dict)
        avg_cost += c/total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

# get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
