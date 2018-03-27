# -*- coding: utf-8 -*-
"""
@author: hannah syrek
This script implements a convolutional neural network to classify the data 
of diabetes type 1 patients.
"""
# Imports
import numpy as np
import os
from utilities import *
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt 
#%matplotlib inline

# Prepare data
X_train, labels_train = read_data(data_path = "./Data/train_set.csv")
X_test, labels_test = read_data(data_path = "./Data/test_set.csv")

# Normalize
X_train, X_test = standardize(X_train, X_test)
X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)

lab_tr = change_label(lab_tr)
lab_vld = change_label(lab_vld)
labels_train = change_label(labels_train)
labels_test = change_label(labels_test)

# One-hot encoding   
y_tr = one_hot(lab_tr)
y_vld = one_hot(lab_vld)
y_test = one_hot(labels_test)

# Hyperparameters
batch_size = 100
seq_len = 20
learning_rate = 0.0001
epochs = 1000
n_classes = 4
n_channels = 20

# Construct the graph
graph = tf.Graph()

# Construct placeholders
with graph.as_default():
    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
    labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
    
# Build Convolutional Layer
with graph.as_default():
    #(batch,20,1) --> (batch,10,2)
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=2, kernel_size=2,strides=1, padding='same', activation = tf.nn.relu)    
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
    #(batch,10,2) --> (batch,5,4)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=4, kernel_size=2,strides=1, padding='same', activation = tf.nn.relu)    
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
# Flatten and pass to the classifier
with graph.as_default():
    flat = tf.reshape(max_pool_2, (-1,5*4))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    # Predictions
    logits = tf.layers.dense(flat, n_classes)
    # Cost function and  Adam optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# Train the network
#if(os.path.exist('checkpoint-cnn')== False):
   #!(mkdir checkpoints-cnn)
    
validation_acc = []
validation_loss = []

train_acc = []
train_loss = []
with graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    
    # Loop over epochs
    for e in range(epochs):
        # Loop over batches
        for x,y in get_batches(X_tr, y_tr,batch_size):
            feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)
            # Print at each 5 iters
            if(iteration % 5 == 0):
                print("Epoch: {}/{}".format(e,epochs),"Iteration: {:d}".format(iteration), "Train loss: {:6f}".format(loss), "Train acc: {:.6f}".format(acc))
            # Compute validation loss at every 10 iterations
            if(iteration%10 == 0):
                val_acc_ = []
                val_loss_ = []
                for x_v, y_v in get_batches(X_vld, y_vld, batch_size):
                    feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}
                    # Loss
                    loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)
                    val_acc_.append(acc_v)
                    val_loss_.append(loss_v)
                    
                # Print info
                print("Epoch: {}/{}".format(e,epochs),"Iteration: {:d}".format(iteration), "Validation loss: {:6f}".format(np.mean(val_loss_)), "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                
                #Store
                validation_acc.append(np.mean(val_acc_))
                validation_loss.append(np.mean(val_loss_))
            # Iterate
            iteration += 1
    saver.save(sess, "checkpoints-cnn/dia.ckpt")
    
# Plot training and test loss
t = np.arange(iteration-1)
plt.figure(figsize = (6,6))
plt.plot(t, np.array(train_loss), 'r-', t[t % 10 ==0], np.array(validation_loss), 'b*')
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.legend(['train','validation'], loc='upper right')
plt.show()

# PLot Accuracy 
plt.figure(figsize = (6,6))  
plt.plot(t, np.array(train_acc), 'r-', t[t % 10 ==0], validation_acc, 'b*')
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.legend(['train','validation'], loc='upper right')
plt.show()
    
    
    
    
   
   
   
