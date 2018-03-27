#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 11:13:38 2018
@author: hannah syrek
Script that implements the needed parameters and utilities to classify data 
with a convolutional neural network.
"""
# Imports
import pandas as pd 
import numpy as np
import os

'''
'''
def read_data(data_path):    
    # Fixed parameters
    n_class = 4
    n_steps = 20
    n_channels= 1
    labels = []
    data = np.zeros((800,20))    
    # Read dataset
    samples = np.genfromtxt(data_path, delimiter = ",", skip_header = 1)
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
'''
def standardize(train,test):
    X_train=(train-np.mean(train,axis=0)[None,:,0])/np.std(train,axis=0)[None,:,0]
    X_test=(test-np.mean(test,axis=0)[None,:,0])/np.std(test,axis=0)[None,:,0]
    
    return X_train, X_test

'''
'''
def change_label(lab):
    new_lab = lab.astype(int)
    for i in range(0,len(new_lab)):
        if(new_lab[i]==4):
            new_lab[i]=3
        elif(new_lab[i]==6):
            new_lab[i]=4
    return new_lab

'''
'''
def one_hot(labels, n_class=4):
    expansion = np.eye(n_class)
    print (expansion)
    y = expansion[:,labels-1].T
    print (y)
    assert y.shape[1] == n_class, "Wrong number of labels!"   
    return y

'''
'''
def get_batches(X,y,batch_size = 50):
    n_batches = len(X)//batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
    # Loop over batches and yield
    for b in range(0,len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
    

        
        