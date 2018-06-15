# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018
@author: Hannah syrek

This script implements the Classic and Derivative DTW algorithm to calculate 
the distances between time series and classifies them with 1-Nearest-Neighbour.
"""

# Needed Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *
import time


'''
Implements a faster version of dynamic time wraping, includes w, a windows size. 
Only mappings within this window size are considered which speeds up the inner
loop of the computations. Good size for w seems to be w = 50.
'''
def DTWDistanceFast(s1, s2,w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
      for j in range(-1,len(s2)):
         DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
      for j in range(max(0, i-w), min(len(s2), i+w)):
         dist= (s1[i]-s2[j])**2
         DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
 
    
'''
Implements LB_Keogh lower bound of dynamic time wraping to speed up the computations.
Good size for r seems to be r=10.
'''
def LB_Keogh(s1,s2,r):
    LB_sum=0  
    for ind,i in enumerate(s1):
        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2
    return np.sqrt(LB_sum)


'''
Implements the k-NearestNeighbor classification to determine the similarity 
between time series, iterate point by point over the hole dataset to find the 
progression of the categories.
'''
def knnDynamic(train,test,w):
    predictions=[]
    dists = []
    start = time.time()
    # The next four lines has to be uncommended for examinations on a test sets, 
    #resort real-world data 
    overlap_timeserie = np.zeros((len(test)-(ts_length-1),ts_length))
    #save all time series with an overlap of 95 % in a new matrix 
    for i in range(0,len(test)-(ts_length-1)):
        overlap_timeserie[i][:] = test[i:ts_length+i]
    test = overlap_timeserie   
    # categorize all time series 
    for ind,i in enumerate(test):        
        min_dist=float('inf')
        closest_seq=[]
        print ind
        #print timeit.Timer(ind)
        for j in train:
            # For the derivation of the datapoints, substitute the following two lines with:
            if LB_Keogh(derive(i[:]),derive(j[:-1]),10)<min_dist:
                dist=DTWDistanceFast(derive(i[:]),derive(j[:-1]),w)
           # if LB_Keogh(i[:],j[:-1],10)<min_dist:
            #    dist=DTWDistanceFast(i[:],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist     
                    closest_seq=j
        # Assign the best 75 % of the time series to the respective class
        predictions.append(closest_seq[-1])  
        dists.append(min_dist) 
    # Compute particular threshold for every class    
    print ("total time", time.time() -start)

    threshold_vec = np.zeros(len(dists))
    dist_data = np.concatenate((np.array([predictions]).T, np.array([dists]).T, np.array([threshold_vec]).T), axis = 1)    
    _class = 1 
    while(_class < 7):
        dist_vec = []
        for i in dist_data: 
            if(i[0]==_class):
                dist_vec.append(i[1])
        # Take only the best 75 percent of the assigned curves
        sort_dist_vec = np.sort(dist_vec)
        _threshold = sort_dist_vec[int((len(sort_dist_vec)*0.5)-1)]
        for j in dist_data:
            if(j[0]==_class):
                j[-1] = _threshold     
        _class += 1
        if(_class == 2 or _class == 5):
            _class +=1
            if(_class == 3):
                _class +=1 
    # Check if distance is bigger than particular threshold, in this case, assgin the residue class
    #for i in dist_data:
        #if(i[1]>i[2]):
           # i[0] = 5.0
    cat_data = np.concatenate((np.array(test), np.array(dist_data)), axis = 1)                                
    #attention: the data includes repetitions of the assigned curves-> use skipRepetitions          
    df = pd.DataFrame(cat_data)
    df.to_csv("Data/DDTW_labeled.csv", index=False)     
    return cat_data


 
print knnDynamic(trainset,labeled_set, 50)
#==============================================================================
#plot the results to visualize the found patterns
#==============================================================================
reload(sys)  
sys.setdefaultencoding('utf8')
plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, ts_length-1, 10, 400])
plt.ylabel('blood glucose content (mg/dL)')
plt.xlabel('timesteps')
plt.show()    
    
    
    
    
    
    
