# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:21:21 2018
@author: hannah syrek

This script implements the Feature-based DTW algorithm to calculate the distances between time
series and classifies them with 1-Nearest-Neighbour.
"""
#Imports
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
    loc_Feat1 = local_Feature(s1)
    loc_Feat2 = local_Feature(s2)
    glo_Feat1 = global_Feature(s1)
    glo_Feat2 = global_Feature(s2)
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
      for j in range(-1,len(s2)):
         DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
      for j in range(max(0, i-w), min(len(s2), i+w)):
         #compute distances according to the definition of method 1 in [Adaptive 
         #Feature Based Dynamic Time Warping, Xie et al.]
         dist_local = abs(loc_Feat1[i][0]-loc_Feat2[j][0]) + abs(loc_Feat1[i][1]-loc_Feat2[j][1])
         dist_global = abs(glo_Feat1[i][0]-glo_Feat2[j][0]) + abs(glo_Feat1[i][1]-glo_Feat2[j][1])
         dist= dist_local + dist_global
         DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
 
    
'''
Implements LB_Keogh lower bound of dynamic time wraping to speed up the computations.
Good size for r seems to be r=10.
'''
def LB_Keogh(s1,s2,r):
    LB_sum=0  
    loc_Feat1 = local_Feature(s1)
    loc_Feat2 = local_Feature(s2)
    glo_Feat1 = global_Feature(s1)
    glo_Feat2 = global_Feature(s2)
    for ind,i in enumerate(s1):         
        part = s2[(ind-r if ind-r>=0 else 0):(ind+r)]
        lower_bound_vec = [i for i, j in enumerate(part) if j == min(part)]
        upper_bound_vec = [i for i, j in enumerate(part) if j == max(part)] 
        if (ind-r>=0):
            lower_bound = lower_bound_vec[0] + r      
            upper_bound = upper_bound_vec[0] + r
        else:
            lower_bound = lower_bound_vec[0]
            upper_bound = upper_bound_vec[0]
            #compute distances according to the definition of method 1 in [Adaptive 
            #Feature Based Dynamic Time Warping, Xie et al.]
            if i>s2[upper_bound]:
                dist_local = abs(loc_Feat1[ind][0]-loc_Feat2[upper_bound][0])+abs(loc_Feat1[ind][1]-loc_Feat2[upper_bound][1])
                dist_global = abs(glo_Feat1[ind][0]-glo_Feat2[upper_bound][0])+abs(glo_Feat1[ind][1]-glo_Feat2[upper_bound][1])
                LB_sum= LB_sum + dist_local + dist_global

            elif i<s2[lower_bound]:
                dist_local = abs(loc_Feat1[ind][0]-loc_Feat2[lower_bound][0])+abs(loc_Feat1[ind][1]-loc_Feat2[lower_bound][1])
                dist_global = abs(glo_Feat1[ind][0]-glo_Feat2[lower_bound][0])+abs(glo_Feat1[ind][1]-glo_Feat2[lower_bound][1])
                LB_sum= LB_sum + dist_local + dist_global

    return np.sqrt(LB_sum)



'''
Implements the k-NearestNeighbor classification to determine the similarity 
between time series, iterate point by point over the hole dataset to find the 
progression of the categories.
'''
def knn_Featurebased(train,test,w):
    predictions=[]
    dists = []
    start = time.time()
#    dyn_timeserie = np.zeros((len(test)-(ts_length-1),ts_length))
#    #save all possible 9716 time series in a new matrix to iterate over all 
#    for i in range(0,len(test)-(ts_length-1)):
#        dyn_timeserie[i][:] = test[i:ts_length+i]
#    test = dyn_timeserie   
    #categorize all time series 
    for ind,i in enumerate(test):        
        min_dist=float('inf')
        closest_seq=[]
        print ind
        for j in train:
            if LB_Keogh((i[:]),(j[:-1]),10)<min_dist:
                dist=DTWDistanceFast((i[:]),(j[:-1]),w)
                if dist<min_dist:
                    min_dist=dist     
                    closest_seq=j
        #assign all time series to the class with the nearest distance
        predictions.append(closest_seq[-1])  
        dists.append(min_dist)  
    print ("total time", time.time() -start)
    threshold_vec = np.zeros(len(dists))
    dist_data = np.concatenate((np.array([predictions]).T, np.array([dists]).T, np.array([threshold_vec]).T), axis = 1)    
    _class = 1    
    while(_class < 7):
        dist_vec = []
        for i in dist_data: 
            if(i[0]==_class):
                dist_vec.append(i[1])
        # Take only the best 25 percent of the assigned curves
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
     # Check if distance is higher than the particular threshold, assgin to residue class
   # for i in dist_data:
       # if(i[1]>i[2]):
          #  i[0] = 5.0
    cat_data = np.concatenate((np.array(test), np.array(dist_data)), axis = 1)                                
    #attention: the data includes repetitions of the assigned curves-> use skipRepetitions
    df = pd.DataFrame(cat_data)
    df.to_csv("Data/fbdtw_set7_anova.csv",  index=False)     
    return cat_data



print knn_Featurebased(trainset,labeled_set, 50)


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

