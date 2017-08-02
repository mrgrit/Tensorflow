import tensorflow as tf

X = [1,2,3]
Y = [1,2,3]

# W에 5.0 입력
W = tf.Variable(-3.0)

hypothesis = W * X
# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis -Y))
#Minimize : Gradient Descent Algorithm
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):    
    print(step, sess.run(W))    
    sess.run(train)
