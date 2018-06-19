import numpy as np

class NeuralNetwork(object):       
    def __init__(self):
#inputlayersize = 5
#hiddenlayersize = 3
#outputlayersize = 1

#Win∈R5×3
        self.Win = np.matrix([[0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01],
                              [0.01, 0.01, 0.01]])
#Wout∈R3×1
        self.Wout = np.matrix([[0.02], [0.02], [0.02]])
#bin∈R3        
        self.Bin = np.matrix([1.0, 1.0, 1.0])
#bout∈R1        
        self.Bout = np.matrix([1.0])
        
    def forwardPropogation(self, X):
        self.z1 = np.dot(X, self.Win)    
#        print(self.z1)
        self.h = self.relu_activation(self.z1 + self.Bin)
#        print(self.h)
        self.z2 = np.dot(self.h,self.Wout)
#        print(self.z2)
        o = self.sigmoid_activation(self.z2 + self.Bout)
        print(o)
        return o
        
    def relu_activation(self,z):    
        return np.maximum(z, 0)
    
    def sigmoid_activation(self,h):
        return 1/(1+np.exp(-h))


X = [1,2,3,4,5]
NN = NeuralNetwork()
o = NN.forwardPropogation(X)
