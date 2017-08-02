import tensorflow as tf
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# placeholder는 동적으로 변수를 받을 때 사용
# shape 는 배열의 형태를 뜻한다.
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

# hypothesis : 공식을 돌려 나올 가정의 값
hypothesis = X * W +b

# 비용 : (가정의 값 - 실제 값) 제곱 의 평균
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # feed_dict를 통해 runtime때 변수를 넣음
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:[1,2,3,4,5], Y:[2.1,2.2,3.3,5,6.1]})
    if step %20 == 0:
        print(step, cost_val, W_val, b_val)

# 예측
print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5,3.5]}))


                
