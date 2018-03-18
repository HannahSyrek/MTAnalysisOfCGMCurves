# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:35:12 2018
@author: hannah syrek
This script implements the Adaptive Feature-based DTW algorithm to calculate the distances between time
series and classifies them with 1-Nearest-Neighbour.
"""
#Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *

#set wi and wj for the first iteration
w_i = 1
w_j = 0

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
         #weighted local and global distance         
         dist= (w_i *dist_local) + (w_j *dist_global)
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
                #weighted local and global distance                   
                LB_sum= LB_sum + (w_i *dist_local) + (w_j *dist_global)

            elif i<s2[lower_bound]:
                dist_local = abs(loc_Feat1[ind][0]-loc_Feat2[lower_bound][0])+abs(loc_Feat1[ind][1]-loc_Feat2[lower_bound][1])
                dist_global = abs(glo_Feat1[ind][0]-glo_Feat2[lower_bound][0])+abs(glo_Feat1[ind][1]-glo_Feat2[lower_bound][1])
                #weighted local and global distance                 
                LB_sum= LB_sum + (w_i *dist_local) + (w_j *dist_global)

    return np.sqrt(LB_sum)


'''
Implements the k-NearestNeighbor classification to determine the similarity 
between time series, iterate point by point over the hole dataset to find the 
progression of the categories.
'''
def knn_AdaptiveFeaturebased(train,test,w):
    predictions=[]
    weights = weighting_Algo(train)
    w_i = weights[0]
    w_j = weights[1]
    dyn_timeserie = np.zeros((9716,ts_length))
    #save all possible time series in a new matrix to iterate over all 
    for i in range(0,len(test)-(ts_length-1)):
        dyn_timeserie[i][:] = test[i:ts_length+i]
    test = dyn_timeserie   
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
        #assign all time series with a higher distance as 30 to the rest catgeory
        print min_dist
        if(min_dist>16):
            predictions.append(5.0)
        else:
            predictions.append(closest_seq[-1])                     
    #produce the categorized dataset: catdata
    #attention: the data includes repetitions of the assigned curves-> use skipRepetitions
    cat_data = np.concatenate((np.array(test), np.array([predictions]).T), axis = 1)  
    df = pd.DataFrame(cat_data)
    df.to_csv("Data/catdatasetAdaptiveFeaturebased16400.csv",  index=False)     
    return cat_data

    
'''
Method to calculate the distance between two time series, componentwise and 
depending on local and global feature of the particular time serie that were 
weighted by the weighting algo.
'''
def AFB_Distance(train,test,w):
    dists=[]
    for ind,i in enumerate(test):        
        min_dist=float('inf')
        print ind
        for j in train:
            if LB_Keogh((i[:-1]),(j[:-1]),10)<min_dist:
                dist=DTWDistanceFast((i[:-1]),(j[:-1]),w)
                if dist<min_dist:
                    min_dist=dist     
        dists.append(min_dist)
    dist_data = np.concatenate((np.array(test), np.array([dists]).T), axis = 1)  
    return dist_data
    
    

'''
Method to compute weights depending on the in-class range of the particular 
samples of the class to weight the local and the global distance to get a better 
classification accuracy. A discription of the algorithm is given in: [Xie, Y., 
& Wiltgen, B. (2010). Adaptive feature based dynamic time warping. International
Journal of Computer Science and Network Security, 10(1), 264-273.]
'''
def weighting_Algo(trainset):
    w=[]
    for i in range(0,2):
        if(i==0):
            w_i=1
            w_j=0
        else:
            w_i=0
            w_j=1            
        dists = AFB_Distance(trainset,trainset,50)
        class_ = 1  
        class_dists=[]
        num_same_classes=[]        
        num_diff_classes = []        
        while(class_<7):
            num_same = 0
            num_diff = 0
            for S_x in dists:
                if(S_x[-2]==class_):
                    num_same +=1
                    class_dists.append(S_x[-1])
            max_dist =max(class_dists)
            num_same_classes.append(num_same) 
            for S_x in dists:
                if(S_x[-2]!=class_ and S_x[-1]<=max_dist):
                    num_diff+=1
            num_diff_classes.append(num_diff)            
            class_ +=1
            if(class_==3 or class_==5):
                class_+=1
        w_i = sum(np.array(num_same_classes) - np.array(num_diff_classes))
        w.append(w_i)  
    return normalize(w[0],w[1])

'''
Method to normalize the weights w1 and w2 as it is introduced in [Xie, Y.,
& Wiltgen, B. (2010). Adaptive feature based dynamic time warping. 
International Journal of Computer Science and Network Security, 10(1), 264-273.]
'''
def normalize(w1,w2):
    if(w1>0 and w2>0):
        w1=w1/float(w1+w2)
        w2=w2/float(w1+w2)
    elif(w1>0 and w2<=0):
        w1=1 
        w2=0
    elif(w1<=0 and w2>0):
        w1=0
        w2=1
    elif(w1<0 and w2<0):
        w1=-w2/float(-(w1+w2))
        w2=-w1/float(-(w1+w2))
    else:
        w1=w2=0.5
    return [w1, w2]



realdata = np.array(skipmissingdata(realdata))
#realdata.resize(486,20) 
#print knn_AdaptiveFeaturebased(train_set,realdata, 50)
print [plotCategories(6)]#,plotCategories(2),plotCategories(4),plotCategories(5),plotCategories(6)]



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



