import tensorflow as tf
import numpy as np

x_data = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# add scope for better graph hierarchy
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2,10]), name = 'weight1')
    b1 = tf.Variable(tf.random_normal([10]), name = 'bias1')
    # layer 1의 임시 hypothesis를 구한다.
    layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    
    # 맨 마지막 layer의 output은 항상 1이다.
    W2 = tf.Variable(tf.random_normal([10,1]), name = 'weight2')
    b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
    # 그다음 layer hypothesis의 X값은 이전 layer의 y값이 들어간다.
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("cost") as scope:
    # Cross Entropy
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    # why the cost values are scalar ??
    cost_summ = tf.summary.scalar("cost",cost)

with tf.name_scope("train") as scope : 
    train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
    # empty train's summary
 
#True if hypothesis > 0.5 else Flase
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))
#WTF ??
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    # tensorboard --logdir = ./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        summary, _ = sess.run([merged_summary, train],feed_dict={X:x_data, Y:y_data})
        writer.add_summary(summary, global_setp = step)
        # sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run([W1,W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

    

                          
