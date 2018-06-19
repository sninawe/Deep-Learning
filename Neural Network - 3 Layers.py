class NeuralNetwork(object):

    def forwardPropogation(self, X, Win, Wout, Bin, Bout):
        """Forward propagation"""
        h = tf.nn.relu(tf.add(tf.matmul(X, Win), Bin))
        o = tf.nn.sigmoid(tf.add(tf.matmul(h, Wout), Bout))
        return o
    
    def weight_init(self,shape):
        """ Weights initialization """
        weights = tf.random_normal(shape=shape, stddev=0.1)
        return tf.Variable(weights)

    def tensor(self,X,Y):
    # Number of input nodes    
        features = X.shape[1]
    # Number of output nodes
        labels = Y.shape[1]
    # Number of hidden nodes
        hidden_nodes = 3
    # TF Placeholders for the inputs and outputs
        tx = tf.placeholder(tf.float32, shape=(None, features))
        ty = tf.placeholder(tf.float32, shape=(None, labels))

    # Weight initializations
        tW1 = self.weight_init(shape=(features,hidden_nodes))
        tW2 = self.weight_init(shape=(hidden_nodes,labels))
        
    # Bias initializations
        tB1 = tf.Variable(tf.random_normal([3],stddev=0.1))
        tB2 = tf.Variable(tf.random_normal([1],stddev=0.1))

    # Forward propagation
        o = self.forwardPropogation(tx, tW1, tW2, tB1, tB2)

    # Backward Propagation
        loss = tf.nn.l2_loss(o - ty)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_step = optimizer.minimize(loss)

    #initializer
        init = tf.global_variables_initializer()

    #run the graph as session
        with tf.Session() as sess:
            sess.run(init) # reset values to wrong
    # training loop
            for i in range(1000):
                sess.run(train_step, {tx:X, ty:Y})

    # evaluate training accuracy
            curr_Win, curr_Bin, curr_Wout, curr_Bout,curr_o = sess.run([tW1, tW2, tB1, tB2,o], {tx:X, ty:Y})
            print("Win: %s bin: %s Wout: %s Bout: %so: %s"%(curr_Win, curr_Bin, curr_Wout, curr_Bout,curr_o))
    
    
X = np.matrix([1,2,3,4,5])
Y = np.matrix([0.3])
NN = NeuralNetwork()
o = NN.tensor(X,Y)    