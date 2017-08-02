import tensorflow as tf

# 학습 데이터
x_data = [[1, 2, 1], [1, 3, 2], [1, 3, 4], [1, 5, 5], [1, 7, 5], [1, 2, 5], [1, 6, 6], [1, 7, 7]]
# one_hot
y_data = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]
# Evaluation our model using this test dataset
x_test = [[2, 1, 1], [3, 1, 2], [3, 3, 4]]
y_test = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]

X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])
W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

# 학습 모델
hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)
# Cross Entropy                        
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 테스트 및 평가 모델
# argmax : (예측값, 실제값) 으로 one hot위치를 알려줌
prediction = tf.arg_max(hypothesis,1)
is_correct = tf.equal(prediction, tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph , 학습 시작 => 학습데이터 사용
# with => 파일, 세션, socket ...의  자동 close
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(201) :
        cost_val, W_val, _ = sess.run([cost, W, optimizer], feed_dict = {X: x_data, Y:y_data})
        print(step, cost_val, W_val)
    # 학습 끝
    # predict => Test 데이터 사용
    print("Prediction: " , sess.run(prediction, feed_dict={X : x_test}))
    # Calculate the accuracy
    print("Accuracy: " , sess.run(accuracy, feed_dict = {X : x_test, Y : y_test}))

                           

