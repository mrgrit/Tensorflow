from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img,cmap = 'gray')

#plt.show()

sess = tf.InteractiveSession()

# 1 image, 28 X 28, 1 color
img = img.reshape(-1,28,28,1)
# 5 filters, 3 X 3, 1 color
W1 = tf.Variable(tf.random_normal([3,3,1,5],stddev=0.01))
# 하나의 값으로 만듬, padding 이 SAME이고 strider가 1 X 1 이면 28 X 28 인데, strider가 2 X 2 일때는 14 X 14가 됨
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding = 'SAME')
print(conv2d)
#Tensor("Conv2D:0", shape=(1, 14, 14, 5), dtype=float32)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img) :
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14),cmap='gray')
    #plt.show()

pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
print(pool)
#Tensor("MaxPool:0", shape=(1, 7, 7, 5), dtype=float32)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img) :
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7,7),cmap='gray')
    plt.show()
