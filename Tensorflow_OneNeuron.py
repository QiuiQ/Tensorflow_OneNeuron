import tensorflow as tf
import numpy as np

v1=1
v2=100
y = np.array([[0]])
b = np.array([0])
starter_learning_rate = 0.1

matrixX = tf.placeholder(tf.float32, shape=([None, None]))
matrixY = tf.placeholder(tf.float32, shape=([None, None]))
bb=tf.placeholder(tf.float32, shape=([1]))
weight = tf.Variable(tf.random_normal([v2,v1]))
prediction = tf.nn.xw_plus_b(matrixX,weight,bb)

global_step = tf.Variable(0, trainable=True)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10, 0.96, staircase=True)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(matrixY - prediction), reduction_indices=[1]))
#================================================================================================
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
#train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
#================================================================================================

'''
learning_rate = tf.train.exponential_time_decay(learning_rate, global_step, k)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(matrixY - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
'''
'''
err = tf.multiply(tf.subtract(matrixY, prediction), tf.multiply(prediction, tf.subtract(1.0, prediction))) #4*1 * 4*1
update = tf.assign(weight, tf.add(weight, tf.matmul(tf.transpose(prediction), err))) #4*4 * 4*1 = 1*1
'''
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    count=0
    x =  np.random.randint(2, size=[v1,v2]) #1*10
    x[0][0]=1
    
    q=np.zeros(dtype=np.float32, shape=(v1, v2))
    for j in range(v2):
        q[0][j]=1
    q[0][0]=-50
    y = np.dot(x,q.T)
    sess.run(train_step, feed_dict={matrixX:x, matrixY:y, bb:b})
    
print("=================W=================")
print(sess.run(weight, feed_dict={matrixX:x, matrixY:y, bb:b}))
print("=================Y================")
ans=sess.run(prediction, feed_dict={matrixX:x, matrixY:y, bb:b})
print(ans)
print("=======")
print("===========finish===========")