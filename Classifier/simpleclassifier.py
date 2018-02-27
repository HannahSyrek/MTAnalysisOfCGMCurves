# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018
@author: hannah syrek
This script implements the DTW algorithm to measure the distances between time
series, a simple knn-classification algorithm and k-means to find patterns in
the datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import random


#==============================================================================
#read data from .csv files
#==============================================================================
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1) 
testset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/testset.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
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
between the time series and classify the dataset with real cgm values.
(Takes round about 5 minutes to run.)
'''
def knn(train,test,w):
    predictions=[]
    #categorize die time series
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        print ind
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],10)<min_dist:
                dist=DTWDistanceFast(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
        predictions.append(closest_seq[-1])
    #produce the categorized dataset catdata
    predsT = np.array([predictions]).T
    realdata = np.array(test)
    catdata = np.concatenate((realdata, predsT), axis = 1)       
    categorie = 1
    meandata = np.zeros((4,20))
    #add and divide the time series of the particular categories to produce 
    #the averaged curve of the specific curve
    while(categorie<7):
        count = 0
        for j in range(0,4):
            for i in range(0,len(predictions)):      
                if(catdata[i][-1]==categorie):
                    meandata[j][:] += catdata[i][:-1]
                    count += 1           
            meandata[j][:] /= count
            categorie += 1
            if (categorie==3 or categorie==5):
                categorie += 1
              
    return meandata


'''
Implements the k-means clustering algorithm. Assign data points to cluster (Expectation), 
recalculate centroids of cluster (Maximization).
(Takes round about 15 minutes to run on the entire dataset for 10 iterations.)
'''
def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print counter
        assignments={}
        #Expectation: Assign data points to cluster
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
Skip the missing cgm values in the real data. Missing values were previously 
replaced by -1.
'''
def skipmissingdata(data): 
    newdata = []
    for count, i in enumerate(data):
        if(i == -1):
            count += 1
        else:
            newdata.append(i)            
    return newdata

   
#==============================================================================
#plot the results to visualize the found patterns
#==============================================================================
D1 = skipmissingdata(realdata1)
D1new = np.array(D1) 
D1new.resize(487,20)
D2 = skipmissingdata(realdata1)
D2new = np.array(D2) 
D2new.resize(487,20)
#stack the two realsets together to have a bigger dataset
realdataset = np.vstack((D1new[:,:],D2new[:,:]))
#print realdataset
df = pd.DataFrame(realdataset)
df.to_csv("realdataset.csv",  index=False)
#stack the train- and testset together to have a bigger dataset
#entiredataset = np.vstack((trainset[:,:-1],testset[:,:-1]))

curves =  knn(trainset,realdataset, 50)
for i in curves:
    plt.plot(i)
  
#fill vector u1 and l1 with the upper- and lower boundvalue to draw them into 
#the visualization
l1 = []
u1 = []
for i in range(0, 20):
    l1 = np.append(l1, 70)
    u1 = np.append(u1, 180)
t_lu = np.asarray(range(0,20))
plt.plot(t_lu, u1,'r--' , t_lu, l1, 'r--')
plt.ylabel('glucose content (mg/dL)')
plt.xlabel('timesteps')
plt.show()


#plot the final centroids of the particular cluster 
#centroids = k_means_clust(realdataset,6,15,100)
#for i in centroids:   
    #plt.plot(i)
        


    
    
    
    
    
    
    
    
    
