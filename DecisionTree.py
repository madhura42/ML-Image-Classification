'''
Created on Dec 14, 2019
'''

import pandas as pd 
import numpy as np
import pickle
from pprint import  pprint
import sys

#check if dataset contains only one class
def purity(train):
    classes = np.unique(train[:,0])
    return True if len(classes) == 1 else False

#classify the datapoint
def predict(train):
    unique_classes, count_unique_classes = np.unique(train[:,0],return_counts = True)
    return unique_classes[count_unique_classes.argmax()]

def create_split(train):
    possible_splits = {}
    features = len(train[1,1:])
    for index in range(1,features): 
        possible_splits[index] = []
        unique_values = np.unique(train[:,index])
        for i in range(len(unique_values)):
            if i != 0: 
                split = (unique_values[i] + unique_values[i-1])/2
                possible_splits[index].append(split)
    return possible_splits

#split data on individual column    
def split(train, split_feature, split_value):
    col_val = train[:,split_feature]
    low = train[col_val <= split_value]
    high = train[col_val > split_value]
    return low,high

#calculate entropy 
def entropy(train):
    y = train[:,0]
    unique_classes, count_unique_classes = np.unique(y,return_counts = True)
    prob = count_unique_classes/count_unique_classes.sum()
    return sum(prob * (-np.log2(prob)))
    
    
def _entropy(low,high):
    total_len = len(low) + len(high)
    prob_low = len(low)/total_len
    prob_high = len(high)/total_len
    return (prob_low*entropy(low) + prob_high*entropy(high)) 
        
def find_best_split(train,possible_splits):
    total_entropy = 9999
    for key in possible_splits:
        for value in possible_splits[key]:
            low,high = split(train, split_feature = key, split_value = value)
            entropy =  _entropy(low,high)
            if entropy <= total_entropy:
                total_entropy = entropy
                best_feature,best_split = key,value
    return best_feature, best_split

def build_tree(train, min_samples = 200, count = 0, max_depth = 3): 
    #check for base case 
    if len(train) < min_samples or purity(train): #or count == max_depth:
        return predict(train)
        
    #recursive call to find best split 
    else:
        best_feature, best_split = find_best_split(train,create_split(train))
        low, high = split(train, best_feature, best_split)
        count+=1
        key = "{} <= {}".format(best_feature,best_split)
        tree = {key: []}
        right_tree = build_tree(low, min_samples, count,max_depth)
        left_tree = build_tree(high, min_samples, count,max_depth)
        if right_tree == left_tree: 
            tree= right_tree
        else:
            tree[key].append(right_tree)
            tree[key].append(left_tree)
        return tree
        
def testTree(X, tree):
     q = list(tree.keys())[0]
     feature, op , value   = (q).split(" ")
     feature = int(feature)
     result = tree[q][0] if X[feature] <= float(value) else tree[q][1]
     if not isinstance(result, dict):
        return result
     else:
        residual_tree = result
        return testTree(X, residual_tree)
    
def tree_accuracy(testX,testY,tree):
    count = 0 
    y_pred = []
    for index in range(len(testX)):
        result = testTree(testX[index],tree)
        y_pred.append(result)
        if result == testY[index]:
            count+=1
    print("Accuracy: " , count/len(testX))
    
            
        
    

if __name__ == '__main__':
    train_data = "train-data.txt"
    test_data = "test-data.txt"
    sys.setrecursionlimit(1000)  
    #training data
    train_df = pd.read_csv(train_data, sep=" ", header = None)
    #train_df = train_df.copy()
    train_df[1][train_df[1] == 90] = 1
    train_df[1][train_df[1] == 180] = 2
    train_df[1][train_df[1] == 270] = 3
    
    train = train_df.to_numpy()
    train = train[:,1:]
    y = train[:, 0] # label
    
    test_df = pd.read_csv(test_data, sep=" ", header = None)
 
    test_df[1][test_df[1] == 90] = 1
    test_df[1][test_df[1] == 180] = 2
    test_df[1][test_df[1] == 270] = 3
    
#Convert data into numpy array
    test = test_df.to_numpy()
    test = test[:,1:]
    X_test = test[:, 1:] # features
    y_test = test[:, 0] # label
    
    
    dTree = (build_tree(train = train,min_samples = 20, count = 0,max_depth=5))
    pprint(dTree)
    file = open("tree_model.txt", 'wb')
    pickle.dump(dTree, file)
    file.close()
    tree_accuracy(X_test,y_test,dTree)
    