#!/usr/local/bin/python3

import sys
import numpy as np
from collections import Counter
import pandas as pd
import pickle
from heapq import nsmallest
from sklearn.preprocessing import LabelBinarizer


# =============================================================================
#Decision Tree
# =============================================================================

class DecisionTree():
    #classify the datapoint
    def predict(self,train):
       unique_classes, count_unique_classes = np.unique(train[:,0],return_counts = True)
       return unique_classes[count_unique_classes.argmax()]
    
    def create_split(self,train):
       possible_splits = {}
       features = len(train[1,1:])
       for index in range(1,features-1): 
           possible_splits[index] = []
           #unique_values = np.unique(train[:,index])
           for i in range(1,len(train[:,index])): 
               if train[i][index] == train[i-1][index]:
                   continue
               else:
                   split = (train[i][index] + train[i-1][index])/2
                   possible_splits[index].append(split)
       return possible_splits
    
    #split data on individual column    
    def split(self,train, split_feature, split_value):
       col_val = train[:,split_feature]
       low = train[col_val <= split_value]
       high = train[col_val > split_value]
       return low,high
    
    #calculate entropy 
    def entropy(self,train):
       y = train[:,0]
       unique_classes, count_unique_classes = np.unique(y,return_counts = True)
       prob = count_unique_classes/count_unique_classes.sum()
       return np.sum(prob * (-np.log2(prob)))
       
       
    def _entropy(self,low,high):
       total_len = len(low) + len(high)
       prob_low = len(low)/total_len
       prob_high = len(high)/total_len
       
       return (prob_low*self.entropy(low) + prob_high*self.entropy(high)) 
           
    def find_best_split(self,train,possible_splits):
       total_entropy = 9999
       for key in possible_splits:
           for value in possible_splits[key]:
               low,high = self.split(train, split_feature = key, split_value = value)
               entropy =  self._entropy(low,high)
               if entropy <= total_entropy:
                   total_entropy = entropy
                   best_feature,best_split = key,value
       return best_feature, best_split
    
    def build_tree(self,train, min_samples = 200, count = 0, max_depth = 3): 
       #check for base case 
       classes = np.unique(train[:,0])
       if len(train) < min_samples or len(classes) == 1 or count == max_depth:
           return self.predict(train)
           
       #recursive call to find best split 
       else:
           best_feature, best_split = self.find_best_split(train,self.create_split(train))
           low, high = self.split(train, best_feature, best_split)
           count+=1
           key = "{} <= {}".format(best_feature,best_split)
           tree = {key: []}
           right_tree = self.build_tree(low, min_samples, count,max_depth)
           left_tree = self.build_tree(high, min_samples, count,max_depth)
           if right_tree == left_tree: 
               tree= right_tree
           else:
               tree[key].append(right_tree)
               tree[key].append(left_tree)
           return tree
           
    def testTree(self,X, tree):
        q = list(tree.keys())[0]
        feature, op , value   = (q).split(" ")
        feature = int(feature)
        result = tree[q][0] if X[feature] <= float(value) else tree[q][1]
        if not isinstance(result, dict):
           return result
        else:
           residual_tree = result
           return self.testTree(X, residual_tree)
       
    def tree_accuracy(self,testX,testY,tree):
       count = 0 
       y_pred = []
       for index in range(len(testX)):
           result = self.testTree(testX[index],tree)
           y_pred.append(result)
           if result == testY[index]:
               count+=1
       print("Accuracy: " , count/len(testX))



# =============================================================================
# K Nearest Neighbors
# =============================================================================

class nearest():
    def __init__(self, k=5):
        self.k = k
        self.trainX = []
        self.trainY = []

    def train(self, X, y):
        self.trainX = X
        self.trainY = y
                
    
    def test(self, testX, testY):
        
        y_pred = []
        c = 0 
        for row in range(len(testX)):
            distance = self.cal_distance(testX, testY)
            #distance = self.cal_distance(testX[row])
            k_votes = nsmallest(self.k, distance, key=lambda x: x[0])
            result = ((Counter(k_votes).most_common(1))[0][0])[1]
            y_pred.append(result)
            if result  == testY[row]:
                c+=1
                
        accuracy = c/len(testX)
        
        return y_pred, accuracy

        
     # euclidean distance calculation 
    def cal_distance(self,testX,testY):
        distance_list = []
        for row in range(len(self.trainX)):
            distance_list.append((np.linalg.norm(self.trainX[row] - testX),self.trainY[row]))
        return distance_list
    
    def cosine_similiarity(self, testX):
        distance_list = []
        for row in range(len(self.trainX)):
            distance = (testX @ testX.T +  np.dot(self.trainX[row], self.trainX[row].T)) - 2*(np.dot(testX.T,self.trainX[row]))
            distance_list.append((distance,self.trainY[row]))
        return distance_list

# =============================================================================
# Neural Network    
# =============================================================================

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
        
        actual = Y_t.argmax(0)
        m = len(actual)
        predicted = final.argmax(0)
        incorrect = np.count_nonzero(actual - predicted)
        return predicted, round((m - incorrect) / m, 4) * 100

def transform(Y):
#    oh = OneHotEncoder()
#    oh.fit(Y)
    lb = LabelBinarizer()
    lb.fit(Y)
    return lb

# =============================================================================
# 
# =============================================================================

if __name__ == "__main__":
    
    # task = "test"
    # fname = "test-data.txt"
    # model_file = "nearest_model.txt"
    # model = "nearest"
    
    task, fname, model_file, model = sys.argv[1:]
  
    if task == "train":
        
        #Reading training dataset
        train_df = pd.read_csv(fname, sep=" ", header = None)
        train_df.loc[1, train_df[1] == 90] = 1
        train_df.loc[1, train_df[1] == 180] = 2
        train_df.loc[1, train_df[1] == 270] = 3
        
        #Convert data into numpy array
        train = train_df.to_numpy()
        train = train[:,1:]
        X = train[:,1:]
        y = train[:, 0] # label
        
        if model == "nearest":
            knn = nearest(k=9)
            knn.train(X,y)
            
            file = open(model_file, 'wb')
            pickle.dump(knn, file)
            file.close()
            #knn.test(X_test,y_test) #test
    
        elif model == "tree":
            d = DecisionTree()
            dTree = d.build_tree(train = train,min_samples = 20, count = 0,max_depth=5)
            file = open(model_file, 'wb')
            pickle.dump(dTree, file)
            file.close()
            
            #d.tree_accuracy(X_test,y_test,dTree) #test
        
        elif model == "nnet" or model == "best":
            X = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
            Y = np.loadtxt(fname, usecols=1, dtype=int)
            
            alpha = 1
            iterations = 5000
            lamb = 0.5
            keep_probability = 0.7
            nnet = NeuralNet(alpha=alpha, iterations=iterations, lamb=lamb, keep_probability= keep_probability) 
            
            lb = transform(Y)
            Y_lb = lb.transform(Y)
            nnet.train(X.T, Y_lb.T)
            
            models = (lb, nnet)
            file = open(model_file, 'wb')
            pickle.dump(models, file)
            file.close()
    
    elif task == "test":
        
        #Reading training dataset
        test_df = pd.read_csv(fname, sep=" ", header = None)
        test_df.loc[1, test_df[1] == 90] = 1
        test_df.loc[1, test_df[1] == 180] = 2
        test_df.loc[1, test_df[1] == 270] = 3
    
        #Convert data into numpy array
        test = test_df.to_numpy()
        test = test[:,1:]
        X_test = test[:, 1:] # features
        y_test = test[:, 0] # label
        
        if model == "nearest":
            knn = pickle.load(open(model_file, "rb"))
            #knn = nearest(k=9)
            _, accuracy = knn.test(X_test, y_test) #test
            print(accuracy)
        
        elif model == "tree":
            d = DecisionTree()
            dTree = pickle.load(open(model_file, "rb"))
            d.tree_accuracy(X_test,y_test,dTree) #test
        
        elif model == "nnet" or model == "best":
            X_test = np.loadtxt(fname, usecols=range(2, 194), dtype=int)
            Y_test = np.loadtxt(fname, usecols=1, dtype=int)
            
            lb, nnet = pickle.load(open(model_file, "rb"))
            
            Y_lb_test = lb.transform(Y_test)
            pred, score = nnet.test(X_test.T, Y_lb_test.T)
            print("Accuracy for test data = ",score)