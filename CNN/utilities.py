#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:13:38 2018
@author: Hannah Syrek
Script that implements the needed parameters and utilities to classify data 
with a convolutional neural network.
"""

# Imports
import pandas as pd 
import numpy as np
import os

'''
Method to read data and cast them into the needed shape: 
(n_timeseries, seq_len, n_channels)
'''
def read_data(data_path):    
    # Fixed parameters
    n_class = 4
    n_steps = 20
    n_channels= 1
    labels = []      
    # Read dataset
    samples = np.genfromtxt(data_path, delimiter = ",", skip_header = 1)
    print (len(samples))
    data = np.zeros((len(samples),20)) 
    # Read labels
    for i in range(0,len(samples)):
        labels.append(samples[i][20])
    labels = np.array(labels)    
    # Reshape data
    for i in range(0,len(samples)):
        data[i][:] = samples[i][:-1]    
    X = data.reshape((len(labels), n_steps, n_channels))     
    #Returns
    return X, labels

'''
Method to standardize the datasets.
'''
def standardize(train,test):
    X_train=(train-np.mean(train,axis=0)[None,:,0])/np.std(train,axis=0)[None,:,0]
    X_test=(test-np.mean(test,axis=0)[None,:,0])/np.std(test,axis=0)[None,:,0]
    # Returns
    return X_train, X_test

'''
Method to change label.
1 -> 1
4 -> 2
6 -> 3
5 -> 4
'''
def change_label(lab):
    new_lab = lab.astype(int)
    for i in range(0,len(new_lab)):
        if(new_lab[i]==4):
            new_lab[i]=2
        elif(new_lab[i]==6):
            new_lab[i]=3
        elif(new_lab[i]==5):
            new_lab[i]=4
    return new_lab

'''
Rechange class label after training, cause of the one-hot encoding.
0 -> 1
1 -> 2
2 -> 3
3 -> 4
'''
def change_class_label(lab):
    new_lab = lab.astype(int)
    for i in range(0,len(new_lab)):
        if(new_lab[i]==0):
            new_lab[i]=1
        elif(new_lab[i]==1):
            new_lab[i]=2
        elif(new_lab[i]==2):
            new_lab[i]=3
        elif(new_lab[i]==3):
            new_lab[i]=4
    return new_lab

'''
Method to encode data label information in one-hot matrices.
'''
def one_hot(labels, n_class=4):
    expansion = np.eye(n_class)
    y = expansion[:,labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"   
    return y

'''
Method to split n_timeseries in m batches.
'''
def get_batches(X,y,batch_size = 50):
    n_batches = len(X)//batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
    # Loop over batches and yield
    for b in range(0,len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
    

        
        