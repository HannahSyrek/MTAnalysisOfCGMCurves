# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:06:59 2018
@author: hannah syrek
This script implements some needed utilities and parameter to run the classification algorithms.
"""

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt





#Initilize some parameters
ts_length = 20
low_value = 70
up_value = 180
lower_bound = []
upper_bound = []
global w_i
global w_j

for i in range(0, ts_length):
    lower_bound.append(low_value)
    upper_bound.append(up_value)
time_steps = np.asarray(range(0,ts_length))



#read the needed data from .csv files
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1) 
realdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))  
#tempcatdata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/catdatasetDerivations.csv",
 #                         delimiter = ",", dtype = None, skip_header = 1) 
tempdata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/catdatasetAdaptiveFeaturebased15.csv",
                          delimiter = ",", dtype = None, skip_header = 1)
'''
Skip the missing cgm values in the real data. Missing values were previously 
replaced by -1.
'''
def skipmissingdata(data): 
    new_data = []
    for count, i in enumerate(data):
        if(i==-1):
            count+=1
        else:
            new_data.append(i)            
    return new_data
   
 
'''
Method to skip the repetitions in the dynamic categorized data.
'''
def skipRepetitions(data):    
    cat_data = np.zeros((len(data),ts_length+1))
    cat_data[0][:] = data[0][:]
    count = 0
    ind = 1
    while(ind < len(data)):
        if(cat_data[count][-1] != data[ind][-1]):  
            cat_data[count+1][:] = data[ind][:]
            count +=1
            ind +=1
        else:
            ind +=1            
    #df = pd.DataFrame(catdata_dyn)
    #df.to_csv("Data/catdataWithoutRepetitions.csv", index=False)       
    return cat_data

 
'''
Method to plot all assigned curves of the particular category.
'''        
def plotCategories(category): 
    data = skipRepetitions(tempdata)   
    count = 0
    for i in data:
        if(i[-1]==category):
            plt.plot(i)
            count += 1
    return count        


'''
Method to calculate the derivation of a given point, as it is used in
[Keogh, E. J., & Pazzani, M. J. (2001, April). Derivative dynamic time warping.
In Proceedings of the 2001 SIAM International Conference on Data Mining
(pp. 1-11). Society for Industrial and Applied Mathematics]. Except the first 
and the last value of the timeserie. Its derivation is computed dependend only
on the second respectively the penultimate value. 
'''
def derive(ts):
    new_timeserie = np.zeros((len(ts))) 
    #set the first and the last derivation each depending on the follwoing respectively the previous point
    new_timeserie[0] = float(((ts[0]-ts[1]) + (ts[0]-ts[1]))/2)/2 
    new_timeserie[-1] = float(((ts[-1]-ts[-2]) + (ts[-1]-ts[-2]))/2)/2
    #compute the derivation of every point in the time serie
    for i in range(1, len(ts)-1):
        new_timeserie[i] = float(((ts[i]-ts[i-1]) + (ts[i+1]-ts[i-1]))/2)/2        
    return new_timeserie



'''
Method to compute the local feature of a datapoint in the given time serie, as it is introduced in 
[Xie, Y., & Wiltgen, B. (2010). Adaptive feature based dynamic time warping. 
International Journal of Computer Science and Network Security, 10(1), 264-273.]
Except the first and the last value of the timeserie. Its local feature is computed
dependend only on the second respectively the penultimate value.
'''
def local_Feature(ts):
    new_timeserie = np.zeros((len(ts),2))
    #set the first and the last feature vector each depending on the follwoing respectively the previous point
    new_timeserie[:][0] = [(ts[0]-ts[1]), (ts[0]-ts[1])]
    new_timeserie[:][-1] = [(ts[-1]-ts[-2]), (ts[-1]-ts[-2])]
    #compute the feature vector of every point in the timeseries according to the definition of Xie et al.
    for i in range(1, len(ts)-1):
        new_timeserie[:][i] = [(ts[i]-ts[i-1]), (ts[i]-ts[i+1])]
    return new_timeserie


'''
Method to compute the global feature of a datapoint in the given time serie, as it is introduced in 
[Xie, Y., & Wiltgen, B. (2010). Adaptive feature based dynamic time warping. 
International Journal of Computer Science and Network Security, 10(1), 264-273.]
Except the first and the last value of the timeserie. Its global feature is computed
dependend only on the second respectively the penultimate value.
'''
def global_Feature(ts):
    new_timeserie = np.zeros((len(ts),2))
    #set the first and the last feature vector each depending on the follwoing 
    #respectively the previous point, additionaly the second, cause by some
    #access problems to vector elements in connection with division with zero in the case of i=1
    new_timeserie[:][0] = [ ts[0] , ts[0]- ((sum(ts[1:])) / float(len(ts)-1)) ] 
    new_timeserie[:][1] = [ ts[1]-ts[0] , ts[1]- ((sum(ts[2:])) / float(len(ts)-1)) ]
    new_timeserie[:][-1] = [ ts[-1]- ((sum(ts[:-1])) / float(len(ts)-1)) , ts[-1] ] 
    #compute the feature vector of every point in the timeseries according to the definition of Xie et al.       
    for i in range(2, len(ts)-1):
        new_timeserie[:][i] = [ ts[i]- (sum(ts[0:i])) / float(i-1) , 
                                ts[i]- float((sum(ts[i+1:])) / float(len(ts)-i)) ]   
    return new_timeserie



#'''
#Method to claculate the max distances within the particular classes to run the
#weight method.
#'''
#
#def DTWDistance(s1, s2):
#    DTW={}
#    for i in range(-1,len(s1)):
#      for j in range(-1,len(s2)):
#         DTW[(i, j)] = float('inf')
#    DTW[(-1, -1)] = 0
#    for i in range(len(s1)):
#      for j in range(len(s2)):
#         dist= (s1[i]-s2[j])**2
#         DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
#    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
#
#
#'''
#Method to compute the max distance within the classes.
#'''
#def max_Distance(classX, sequence):
#    dists = [-10]
#    for i in enumerate(classX):
#        dists.append(DTW(i,sequence))
#        max_dist = max(dists)
#    return max_dist
#  
#
#        max_class1 =max(dists[0  :99][-1])
#        max_class2 =max(dists[100:199][-1])
#        max_class4 =max(dists[200:299][-1])
#        max_class6 =max(dists[300:399][-1])
#        
#        num_class1 =len(dists[0  :99][-1])
#        num_class2 =len(dists[100:199][-1])
#        num_class4 =len(dists[200:299][-1])
#        num_class6 =len(dists[300:399][-1])
#        
#        
#        if(dists[100:399][-1]<=max_class1):
#            cum1+=1
#        
#        
#        
#        
#        for i in enumerate(dists):
#            for jnd, j in enumerate(dists):
#                jnd +=1
#                if(i[-1]>=j[-1])
#        for time_seq in enumerate(trainset):
#            if(time_seq[-1]==1):
#             class1= np.hstack(class1,time_seq)   
#             max_dist = max_Distance(class1,time_seq)
#             num_class1 =len(class1[0])
#            elif(time_seq[-1]==2):
#             class2= np.hstack(class2,time_seq)
#             max_dist = max_Distance(class2,time_seq)
#            elif(time_seq[-1]==4):
#             class4= np.hstack(class4,time_seq)
#             max_dist = max_Distance(class4)
#            elif(time_seq[-1]==6):
#             class6= np.hstack(class6,time_seq)
#             max_dist = max_Distance(class6, time_seq)
#
#            maxDistclass1 =   
