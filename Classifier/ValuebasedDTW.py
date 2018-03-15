# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018
@author: hannah syrek

This script implements the Classic and Derivative DTW algorithm to calculate 
the distances between time series and classifies them with 1-Nearest-Neighbour.
"""
#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *


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
    dyn_timeserie = np.zeros((9716,ts_length))
    #save all possible 9716 time series in a new matrix to iterate over all 
    for i in range(0,len(test)-(ts_length-1)):
        dyn_timeserie[i][:] = test[i:ts_length+i]
    test = dyn_timeserie   
    #categorize all time series 
    for ind,i in enumerate(test):        
        min_dist=float('inf')
        closest_seq=[]
        print ind
        for j in train:
            #For the derivation of the datapoints, substitute the following two lines with:
            #if LB_Keogh(derive(i[:]),derive(j[:-1]),10)<min_dist:
                #dist=DTWDistanceFast(derive(i[:]),derive(j[:-1]),w)
            if LB_Keogh(derive(i[:]),derive(j[:-1]),10)<min_dist:
                dist=DTWDistanceFast(derive(i[:]),derive(j[:-1]),w)
                if dist<min_dist:
                    min_dist=dist     
                    closest_seq=j
        #assign all time series with a higher distance as 70 respectively 20 to the rest catgeory
        #For the derivation of the datapoints, substitute the following line with:
        #if(min_dist<15):
        if(min_dist>15):
            predictions.append(5.0)
        else:
            predictions.append(closest_seq[-1])                     
    #produce the categorized dataset: catdata
    #attention: the data includes repetitions of the assigned curves-> use skipRepetitions
    cat_data = np.concatenate((np.array(test), np.array([predictions]).T), axis = 1)  
    df = pd.DataFrame(cat_data)
    df.to_csv("Data/catdatasetDerivations.csv",  index=False)     
    return cat_data


 
realdata = np.array(skipmissingdata(realdata))
#print knnDynamic(trainset,realdata, 50)
#print plotCategories(5)# ,plotCategories(2),plotCategories(4),plotCategories(5),plotCategories(6)]



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
    
    
    
    
    
    
