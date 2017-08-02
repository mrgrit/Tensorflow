import tensorflow as tf
import numpy as np

x = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [0], [0], [1]]

learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, 2], name="X-input")
Y = tf.placeholder(tf.float32, [None, 1], name="Y-input")

with tf.name_scope("Layer") as scope:
    W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='weight')
    B = tf.Variable(tf.zeros([1]), name='bias')
    L = tf.sigmoid(tf.matmul(X, W) + B)

with tf.name_scope("Cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(L)+ (1-Y)*tf.log(1-L))
    cost_sum = tf.summary.scalar("Cost", cost)

with tf.name_scope("Train") as scope:
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("Accuracy") as scope:
    predicted = tf.cast(L > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
    accuracy_sum = tf.summary.scalar("Accuracy", accuracy)

init = tf.global_variables_initializer()


with tf.Session() as sess:

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("./logs/and_01")
	writer.add_graph(sess.graph)

	sess.run(init)

	for step in range(100):
	    summary, _ = sess.run([merged, train], feed_dict={X: x, Y: y})
	    writer.add_summary(summary, step)

	    if step % 10 == 0:
	        print(step, sess.run(cost, feed_dict={X: x, Y: y}), sess.run([W]))

	print(sess.run(accuracy, feed_dict={X: x, Y: y}))
