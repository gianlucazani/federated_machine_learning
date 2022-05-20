import sys
import os
import numpy as np
import json


HOST = "127.0.0.1"


class Client:
    def __init__(self):
        self.id = sys.argv[1]
        self.port_no = sys.argv[2]
        self.server_port_no = 6000  # fixed to 6000
        self.optimization_method = sys.argv[3]  # 0 for GD, 1 for Mini-Batch GD
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []


    def run(self):
        self.read_dataset()

    def read_dataset(self):
        train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(self.id) + ".json")
        test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(self.id) + ".json")
        train_data = {}
        test_data = {}
        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            test_data.update(test['user_data']) 
        self.X_train, self.Y_train, self.X_test, self.Y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']

    def softmax(self, X, W):
        Z = X.dot(W)
        e_Z = np.exp(Z)
        A = e_Z / e_Z.sum(axis = 1, keepdims = True)
        return A
    
    def softmax_loss(self, X, y, W):
        A = self.softmax(X,W)
        id0 = range(X.shape[0])
        return -np.mean(np.log(A[id0, y]))

    def softmax_grad(self, X, y, W):
        pred = self.softmax(X,W)  
        xid = range(X.shape[0])   
        pred[xid, y] -= 1        
        return X.T.dot(pred)/X.shape[0]

    def softmax_fit(self, X, y, W, lr = 0.01, nepoches = 2):
        ep = 0 
        loss_hist = [self.softmax_loss(X, y, W)] # store history of loss 
        N = X.shape[0]
        while ep < nepoches: 
            ep += 1 
            W -= lr*self.softmax_grad(X, y, W)
            loss_hist.append(self.softmax_loss(X, y, W))
        return W, loss_hist 

    def accuracy(self, y_pre, y):
        diff = y_pre-y 
        n_zeros = np.count_nonzero(diff==0)
        print(f'correctly predicted: {n_zeros}')
        np.asanyarray(n_zeros)
        accuracy = n_zeros/len(y)
        print(accuracy)
        return accuracy

    def pred(self, W, X):
        A = self.softmax(X,W)
        return np.argmax(A, axis = 1)


client = Client()
client.run()



