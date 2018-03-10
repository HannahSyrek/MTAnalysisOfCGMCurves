# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:06:59 2018
@author: hannah syrek
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
tempcatdata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/catdatasetDerivations.csv",
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
    catdata_dyn = np.zeros((len(data),ts_length+1))
    catdata_dyn[0][:] = tempcatdata[0][:]
    count = 0
    ind = 1
    while(ind < len(data)):
        if(catdata_dyn[count][-1] != data[ind][-1]):  
            catdata_dyn[count+1][:] = data[ind][:]
            count +=1
            ind +=1
        else:
            ind +=1            
    df = pd.DataFrame(catdata_dyn)
    df.to_csv("Data/catdataWithoutRepetitions.csv", index=False)       
    return catdata_dyn

 
'''
Method to plot all assigned curves of the particular category.
'''        
def plotCategories(category): 
    data = skipRepetitions(tempcatdata)    
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
and the last value of the timeserie.  Its derivation is computed dependend only
 from the second respectively the penultimate value. 
'''
def derive(ts):
    first_value = float(((ts[0]-ts[1]) + (ts[0]-ts[1]))/2)/2 
    last_value = float(((ts[-1]-ts[-2]) + (ts[-1]-ts[-2]))/2)/2
    new_timeserie = np.zeros((len(ts)))    
    new_timeserie[0] = first_value
    new_timeserie[-1] = last_value
    for i in range(1, len(ts)-1):
        derivation = float(((ts[i]-ts[i-1]) + (ts[i+1]-ts[i-1]))/2)/2
        new_timeserie[i] = derivation
    return new_timeserie

