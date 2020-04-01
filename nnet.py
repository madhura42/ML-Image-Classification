# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:39:39 2019

"""

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle
np.random.seed(3)

class NeuralNet(object):
    def __init__(self, alpha=1, iterations = 5000, lamb = 0.5, layer_dimension = [192,128,128,4], keep_probability = 0.8):
        self.alpha = alpha
        self.iterations = iterations
        self.lamb = lamb
        self.layer_dimension = layer_dimension
        self.keep_probability = keep_probability
        self.params = {}
        for i in range(1, len(layer_dimension)):
            self.params["weights"+str(i)] = np.random.randn(layer_dimension[i],layer_dimension[i-1])
            self.params["bias"+str(i)] = np.zeros((layer_dimension[i],1))
            
    def sigmoid(self, x):
        denom = 1 + np.exp(-x)
        return 1 / (denom + (1e-10))
    
    def softmax(self, x):
        e_x = np.exp(x-np.max(x))
        sum1 = np.sum(e_x, axis=0, keepdims=True)
        return e_x/sum1
    
    def dropout(self, X, drop_probability):
        #keep_probability = 1- drop_probability
        mask = np.random.rand(X.shape[0],X.shape[1]) < self.keep_probability
        X = X*mask
        X = X/self.keep_probability
        return X, mask
    
    def cross_entropy(self,ZL,Y):
        temp1 = np.sum(np.exp(ZL), axis=0, keepdims=True)
        temp2 = ZL - np.log(temp1)
        temp3 = np.multiply(Y, temp2)
        return - np.sum(temp3)
    
    def compute_cost(self,AL,Y,caches):
        m = Y.shape[1]
        length = len(self.layer_dimension)-1
        ZL = caches[-1][1]
        cross_entropy = self.cross_entropy(ZL,Y)/m
        sum = 0
        for i in range(1,length+1):
            sum += np.sum(np.square(self.params["weights"+str(1)]))
        std_cost = self.lamb*sum/(2*m)
        
        cost = np.squeeze(cross_entropy)+std_cost
        return cost
    
    def derivative_ZL(self, X, Y):
        return X-Y
    
    def activation_function_forward(self, X, weights, bias, activation_function):
        Z = np.dot(weights, X) + bias
        
        if activation_function == "sigmoid":
            x_next = self.sigmoid(Z)
        
        elif activation_function == "softmax":
            x_next = self.softmax(Z)
            
        cache = ((X, weights, bias),Z)   
        return x_next, cache
    
    def forward_propogation_train(self, x_next):
        
        length = len(self.layer_dimension)-1
        caches = []
        for i in range(1,length):
            x = x_next
            weights = self.params["weights" + str(i)]
            bias = self.params["bias" + str(i)]
            x_next, cache = self.activation_function_forward(x, weights, bias, "sigmoid")
            caches.append(cache)
            
        weights = self.params["weights"+ str(length)]
        bias = self.params["bias" + str(length)]
        output, cache = self.activation_function_forward(x_next, weights, bias, "softmax")
        caches.append(cache)
        
        return output, caches
    
    def backward_linear(self, dx, cache, i):
        x, weights, bias = cache
        m = x.shape[1]
        weights = self.params["weights"+str(i)]
        #updating the weights and biases
        dW = (1/m)*np.dot(dx, x.T) + (self.lamb*weights)/m
        db = (1/m)*np.sum(dx, axis=1, keepdims=True)
        dx_prev = np.dot(weights.T,dx)
        
        return dx_prev, dW, db
    
    def backward_activation_function(self, dA, cache, activation, l):
        ((A_prev, weights, bias), Z) = cache
        
        if activation == "sigmoid":
            temp = self.sigmoid(-Z)
            dZ = dA * temp * (1-temp)

        elif activation == "softmax":
            temp = self.softmax(Z)
            dZ = temp - Z

        dA_prev, dW, db = self.backward_linear(dZ, (A_prev, weights, bias), l)
        return dA_prev, dW, db
    
    def backpropogation(self, AL, Y, caches):
        values = {}
        cache_length = len(caches)
        Y = Y.reshape(AL.shape)
        cache = caches[cache_length-1]
        dZL = self.derivative_ZL(AL,Y)
        values["dA"+str(cache_length)], values["dW"+str(cache_length)], values["db"+str(cache_length)] = self.backward_linear(dZL, cache[0], cache_length)
        for i in range(cache_length-1, 0, -1):
            cache = caches[i-1]
            dx_prev_temp, dW_temp, db_temp = self.backward_activation_function(values["dA"+str(i+1)], cache, "sigmoid", i)
            values["dA"+str(i)] = dx_prev_temp
            values["dW"+str(i)] = dW_temp
            values["dA"+str(i)] = db_temp
            
        return values
    
       
    def train(self, trainX, trainY):
        length = len(self.layer_dimension)-1
        costs = []
        for i in range(self.iterations+1):
            final, caches = self.forward_propogation_train(trainX)
            
            cost = self.compute_cost(final,trainY,caches)
            values = self.backpropogation(final,trainY,caches)
            for j in range(1,length+1):
                self.params["weights"+str(j)] = self.params.get("weights"+str(j), 0) - self.alpha*values.get("dW"+str(j), 0)
                self.params["bias"+str(j)] = self.params.get("bias"+str(j), 0) - self.alpha*values.get("db"+str(j), 0)

            if i%500 == 0:    
                costs.append((i, cost))
                _, accuracy = self.test(trainX,trainY)
                print("Iteration", i, "->", "Accuracy = ", accuracy, " Cost = ",cost)
    
#    def test(self, testX, testY):
#        predicted = self.forward_propogation_test(testX)
#        original = testY
#        m = len(original)
#        incorrect = np.count_nonzero(original-predicted)
#        accuracy = round((m - incorrect)/m*100, 4)
#        return predicted, accuracy
    
    def test(self, X_test, y_test):
        X_t = X_test
        Y_t = y_test
        final, caches = self.forward_propogation_train(X_t)
        m = len(actual)
        
        actual = Y_t.argmax(0)
        predicted = final.argmax(0)
        incorrect = np.count_nonzero(actual - predicted)
        return predicted, round((m - incorrect) / m, 4) * 100
    
def transform(Y):
#    oh = OneHotEncoder()
#    oh.fit(Y)
    lb = LabelBinarizer()
    lb.fit(Y)
    return lb
#    return oh

if __name__ == '__main__':
    fname = "train-data.txt"
    image = np.loadtxt(fname, usecols=0, dtype=str)
    X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    trainX, trainY = [], []
    Y = np.loadtxt(fname, usecols=1, dtype=int)
    
    fname = "test-data.txt"
    image = np.loadtxt(fname, usecols=0, dtype=str)
    X_test = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
    testX, testY = [], []
    Y_test = np.loadtxt(fname, usecols=1, dtype=int)
    
    alpha = 1
    iterations = 5000
    lamb = 0.5
    keep_probability = 0.7
    nnet = NeuralNet(alpha=alpha, iterations=iterations, lamb=lamb, keep_probability= keep_probability) 
    lb_train = transform(Y)
    Y_lb = lb_train.transform(Y)
    nnet.train(X.T, Y_lb.T)
    
    lb_test = transform(Y_test)
    Y_lb_test = lb_test.transform(Y_test)
    pred, score = nnet.test(X_test.T, Y_lb_test.T)
    print("Accuracy for test data = ",score)
    
    models = (lb_test, nnet)
    file = open("nnetmodel.txt", 'wb')
    pickle.dump(models, file)
    file.close()
    
