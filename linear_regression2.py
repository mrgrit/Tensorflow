import tensorflow as tf

# 1-1, 2-2, 3-3 인 학습 데이터
x_train = [1,2,3]
y_train = [1,2,3]

# W와 b 를 random 값으로 초기화
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# linear regression 의 수식 : X_train값에 임의의 W와 b를 넣으면 y가 나오도록 함
hypothesis = x_train * W + b

# 비용의 의미 : 선형과 실제 값과의 거리
# 가정의 값에 실제값을 빼서 제곱하고 평균을 구함
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# 경사하강법 : 어떤 곡선을 미분 하여 기울기를 따라 하강하며 최적의 값을 추적
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

#tf의 세션 생성
sess=tf.Session()
# Initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
