import tensorflow as tf

# training data
#prepare dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# define computational graph
# Create the model
x = tf.placeholder(tf.float32, [None, 784])
h = tf.layers.dense(x,30)
y = tf.layers.dense(h,10)

#initializer
init = tf.global_variables_initializer()

# Define loss and optimizer
y_ = tf.placeholder(tf.int64, [None,10])
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=y)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

loss_log = []

#run the graph as session
with tf.Session() as sess:
    sess.run(init) # reset values to wrong
    # training loop
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        loss,_ = sess.run([cross_entropy,train_step], feed_dict={x: batch_xs, y_: batch_ys})
        loss_log.append(loss)

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_result = sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    print("accucracy is ",accuracy_result)