import tensorflow as tf
import numpy as np

# NP Slicing 다시 볼것
xy = np.loadtxt('ais_sample2.csv',delimiter = ',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None,11])
Y = tf.placeholder(tf.float32, shape=[None,1])

# 11 : 들어오는 값(X), 1 : 나가는 값(Y)
W = tf.Variable(tf.random_normal([11,1]), name = 'Weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 행렬 곱 + bias
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 정확도 확인 -> True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

# Launch Graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed = {X:x_data, Y:y_data}
    for step in range(10001):
        sess.run(train, feed_dict=feed)
        if step % 200 == 0:
            print(step, sess.run(cost,feed_dict = feed))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = feed)
    print("\nHypothesis : ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
