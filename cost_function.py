import tensorflow as tf

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros[1])
y = W*x_data + b

# 거리기반 비용함수
# 거리에 제곱, 그 합계에 평균
loss  = tf.reduce_mean(tf.square(y - y_data))

# 경사하강법 알고리즘
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
see.run(init)

for step in range(8) :
    sess.run(train)

print(sess.run(W), sess.run(b))
