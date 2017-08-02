import tensorflow as tf
import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
# wight : 기울기, b : 절편
W= tf.placeholder(tf.float32)
#공식 대로
hypothesis = x * W
#공식 대로
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Tensor flow 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 그래프를 그리기 위한 변수들
W_val = []
cost_val = []

for i in range(-30, 50):
    feed_W = i * 0.1
    # feed_W는 place holder W에 들어가고 이를 대입시켜 코스트를 구함
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    # 연속으로 값을 리스트에 넣어 그래프를 그림
    W_val.append(curr_W)
    cost_val.append(curr_cost)
    

# 그래프에서 수행 할 수록 비용이 점점 줄어들다가
# 최적의 단계를 지나면 다시 늘어남
plt.plot(W_val, cost_val)
plt.show()
