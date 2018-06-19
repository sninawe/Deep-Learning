import numpy as np

class Two2dGaussianData(object):
    '''
    Dataset of two 2d gaussian as a toy binary classification
    '''
    def __init__(self):
        '''
        initialize data with 2000 data points for training, 
        200 data points for validiation, 200 data points for test 
        '''
        N = 2000
        x0 = np.random.randn(N, 2) + np.array([0.9, 0.9])
        x1 = np.random.randn(N, 2) + np.array([-0.9, -0.9])
        self.X = {}
        self.X["train"]=np.vstack((x0[0:1000], x1[1000:2000]))
        self.X["val"]=np.vstack((x0[2000:2200], x1[2000:2200]))
        self.X["test"]=np.vstack((x0[2200:2400], x1[2200:2400]))
        y0=np.zeros(N)#.astype(np.int)
        y1=np.ones(N)#.astype(np.int)
        self.y={}
        self.y["train"]=np.hstack((y0[0:1000], y1[1000:2000]))
        self.y["val"]=np.hstack((y0[2000:2200], y1[2000:2200]))
        self.y["test"]=np.hstack((y0[2200:2400], y1[2200:2400]))
        self.leraning_rate=0.00001
        self.num_iterations=300000
        
    def get_batch(self,batch_size,mode="train"):
        #get random batch
        num_all_data=len(self.X[mode])
        random_indices=np.random.choice(num_all_data, batch_size, replace=False)
        Xbatch=self.X[mode][random_indices]
        ybatch=self.y[mode][random_indices]
        return Xbatch,ybatch

    def loss_func(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit_model(self, Xbatch, ybatch):
        
        # intercept initialization
        b = np.ones((Xbatch.shape[0], 1))
        # Add X and intercept
        Xbatch = np.hstack((Xbatch,b))
        # weights initialization
        self.w = np.zeros(Xbatch.shape[1])
        
        for i in range(self.num_iterations):
            z = np.dot(Xbatch, self.w)
            h = self.sigmoid(z)
            #gradient/partial derivative w.r.t weight
            gradient = np.dot(Xbatch.T, (h - ybatch)) / ybatch.size
            self.w -= self.leraning_rate * gradient
            
            if(i % 10000 == 0):
                z = np.dot(Xbatch, self.w)
                h = self.sigmoid(z)
                print(f'cross entropy loss = {self.loss_func(h, ybatch)} \t')
        #We can use this weight for predicting y in the test set.
        print(self.w)        
    
LR = Two2dGaussianData()
X_train, y_train = LR.get_batch(2000,mode="train")
model = LR.fit_model(X_train, y_train)
