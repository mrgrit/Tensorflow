import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
image = np.array([
[[[1],[2],[3]],
[[4],[5],[6]],
[[7],[8],[9]]]
],dtype = np.float32)


print(image.shape)
plt.imshow(image.reshape(3,3), cmap = 'Greys')
#plt.show()

# print("image:\n",image)
# filter(weight) 생성
weight = tf.constant([
[[[1.]],[[1.]]],
[[[1.]],[[1.]]]
])

print("weight.shape : ", weight.shape)
conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1],padding='VALID')
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ", conv2d_img.shape)
# swapaxes함수는 직관적으로 바꾸고자하는 축 두개 번호를 전달받아 두 축에 대해서 바꾸게 된다.
conv2d_img = np.swapaxes(conv2d_img, 0 ,3)
# enumerate : (index, 항목)형태의 tuple을 생성해줌
for i, one_img in enumerate(conv2d_img) :
    print(one_img.reshape(2,2))
    # subplot : http://pinkwink.kr/972
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')
    #plt.show()

print("image.shape2 : ", image.shape)
weight = tf.constant([
[[[1.]],[[1.]]],
[[[1.]],[[1.]]]
])
print("weight.shape : ",weight.shape)
conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding = 'SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ",conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img,0,3)
for i, one_img in enumerate(conv2d_img):
    # shape : 3 X 3 include padding
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap = 'gray')
    # plt.show()

# 3개의 weight => 3개의 filter
print("image.shape3 : ", image.shape)
weight = tf.constant([
[[[1.,10.,-1.]],[[1.,10.,-1.]]],
[[[1.,10.,-1.]],[[1.,10.,-1.]]]
])
print("weight.shape : ",weight.shape) #(2, 2, 1, 3), 2 X 2 지만 padding이 들어가서 3 X 3가 됨
conv2d = tf.nn.conv2d(image,weight, strides=[1,1,1,1],padding='SAME')
conv2d_img = conv2d.eval()
print("conv2d_img.shape : ", conv2d) # (1, 3, 3, 3)
conv2d_img = np.swapaxes(conv2d_img,0,3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,3,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')
    #plt.show()
"""
[[ 12.  16.   9.]
 [ 24.  28.  15.]
 [ 15.  17.   9.]]
[[ 120.  160.   90.]
 [ 240.  280.  150.]
 [ 150.  170.   90.]]
[[-12. -16.  -9.]
 [-24. -28. -15.]
 [-15. -17.  -9.]]
"""
image = np.array([
[[[4],[3]],
[[2],[1]]]
], dtype = np.float32)
# max plling
pool = tf.nn.max_pool(image,ksize=[1,2,2,1], strides=[1,1,1,1], padding='VALID')
print(pool.shape)
print(pool.eval())

image = np.array([
[[[4],[3]],
[[2],[1]]]
], dtype = np.float32)
# max plling
pool = tf.nn.max_pool(image,ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
print(pool.shape)
print(pool.eval())
