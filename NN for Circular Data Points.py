import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd 

PI=np.pi
PI2=2*PI

def gencircle(rc,rr=0.1,offset=[0,0],num=100,label=0):
    #taken from https://qiita.com/xiangze750/items/1d024c8536d128c3ceae
    c=[]
    for i in range(num):
        r=rc+np.random.uniform(-rr,rr,1)
        th=np.random.uniform(0,PI2,1)
        c.append([r*np.sin(th)+offset[0],r*np.cos(th)+offset[1]])
    return np.c_[np.array(c).reshape(num,2),np.repeat(label,num)]

X=np.r_[gencircle(0.1,0.1,num=1000,label=0),gencircle(0.5,0.1,num=1000,label=1)][:,0:2]

arr1 = np.full((1000, 1), 0, dtype=int)
arr2 = np.full((1000, 1), 1, dtype=int)
Y = np.append(arr1,arr2)

plt.scatter(X[0:1000,0],X[0:1000,1],color="red")
plt.scatter(X[1000:2000,0],X[1000:2000,1],color="blue")

# define computational graph
# Create the model

data = tf.placeholder(tf.float32, [None, 2])
h = tf.layers.dense(data,24,activation=tf.nn.relu)
y = tf.layers.dense(h,1,activation=tf.nn.sigmoid)

#initializer
init = tf.global_variables_initializer()

# Define loss and optimizer
target = tf.placeholder(tf.float32, [None,1])
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=target))
train_step = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

prediction = tf.round(y)
correct = tf.cast(tf.equal(prediction, target), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)

train_acc = []
batch_size = 30
iter_num = 30000
sess = tf.Session()

for epoch in range(iter_num):
    sess.run(init)
    # Generate random batch index
    batch_index = np.random.choice(len(X), size=batch_size)
    batch_train_X = X[batch_index]
    batch_train_y = np.matrix(Y[batch_index]).T
    sess.run(train_step, feed_dict={data: batch_train_X, target: batch_train_y})
    temp_loss = sess.run(loss, feed_dict={data: batch_train_X, target: batch_train_y})
    # convert into a matrix, and the shape of the placeholder to correspond
    temp_train_acc = sess.run(accuracy, feed_dict={data: X, target: np.matrix(Y).T})
    # recode the result
    train_acc.append(temp_train_acc)
    # output
    if (epoch + 1) % 2000 == 0:
        print('epoch: {:4d} train_acc: {:5f}'.format(epoch + 1,temp_train_acc)) 
#    curr_y, curr_target = sess.run([y, target], {data: batch_train_X, target: batch_train_y})
#    print("y: %s target: %s"%(curr_y, curr_target))  
