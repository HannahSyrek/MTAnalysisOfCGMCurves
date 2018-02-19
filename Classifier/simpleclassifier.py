# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018

@author: hannah syrek
This script implements the DTW algorithm to measure the distances between time
series, a simple classification algorithmes them with k-means.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random


#==============================================================================
#read from train- and testset
#==============================================================================
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", 
                         delimiter = ",", dtype = None) 
testset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/testset.csv", 
                        delimiter = ",", dtype = None) 
entiredataset = np.vstack((trainset[:,:-1],testset[:,:-1]))
#print entiredataset

realdata1 = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3)) 
realdata2 = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v10.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))    


'''
Implements a faster version of dynamic time wraping, includes w, a windows size. 
Only mappings within this window size are considered which speeds up the inner
loop of the computations. Good size for w seems to be w = 100.
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
Implements LB_Keogh lower bound of dynamic timewraping to speed up the computations.
Good size for r seems to be r=10
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
Implements the K-NearestNeighbor classification to determine the similarity 
between two time series more precisely two Cgm curves.
Takes round about 30 minutes to run.
'''
def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        #print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],10)<min_dist:
                dist=DTWDistanceFast(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds)

 
'''
Implements the k-means clustering algorithm. Assign data points to cluster(Expectation), 
recalculate centroids of cluster (Maximization).
Takes round about 10 minutes to run.
'''
def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #Expectation: Assign data points to clusters
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,10)<min_dist:
                    cur_dist=DTWDistanceFast(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]
        #Maximization: Recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    return centroids


'''
Skip the missing cgm values in the real data, put the two sets together 
to have a bigger dataset.
'''
def skipmissingdata(data): 
    newdata = []
    for count, i in enumerate(data):
        if(i == -1):
            count += 1
        else:
            newdata.append(i)            
    return newdata

D1 = skipmissingdata(realdata1)
D1new = np.array(D1) 
D1new.resize(487,20)
D2 = skipmissingdata(realdata1)
D2new = np.array(D2) 
D2new.resize(487,20)
realdataset = np.vstack((D1new[:,:],D2new[:,:]))

   
#==============================================================================
#call some methodes and plot the results to visualize the found patterns
#==============================================================================
l1 = []
u1 = []
for i in range(0, 20):
    l1 = np.append(l1, 70)
    u1 = np.append(u1, 180)
t_lu = np.asarray(range(0,20))
 
centroids = k_means_clust(realdataset,4,10,100)
for i in centroids:   
    plt.plot(i)

plt.plot(t_lu, u1,'r--' , t_lu, l1, 'r--')
plt.ylabel('glucose content (mg/dL)')
plt.xlabel('timesteps')
plt.show()

        


    
    
    
    
    
    
    
    
    
