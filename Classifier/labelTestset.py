# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:36:23 2018

@author: hannah syrek
Script to label manually a test dataset to compute the accuracy rate of the implemented classifier.
"""
#Imports
#import os
#os.chdir("/home/hannah/Dokumente/MTAnalysisOfCGMCurves")
from utilities import *
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


realdata = np.array(skipmissingdata(realdata))


all_timeseries = np.zeros((9716,ts_length))
#save all possible 9716 time series in a new matrix to iterate over all 
for i in range(0,len(realdata)-(ts_length-1)):
    all_timeseries[i][:] = realdata[i:ts_length+i]
real_data = all_timeseries
  
df = pd.DataFrame(real_data)
df.to_csv("Data/datasetLabeled.csv",  index=False)   

c_means=np.zeros((4,20))
#trainset.T
#print len(trainset.T[0][0:100])
for i in range(0,20):
    c_means[0][i] = np.mean(trainset.T[i][0:100])
    c_means[1][i] = np.mean(trainset.T[i][100:200])
    c_means[2][i] = np.mean(trainset.T[i][200:300])
    c_means[3][i] = np.mean(trainset.T[i][300:400])
#print c_means
for i in c_means:
    plt.plot(i)
#plt
    #.plot(c_means)
    5
#+5
k=154
for i in range(k,k+1):
    c=real_data[i][:]   
    print real_data[i][:]
    plt.plot(c,'m',label = i)












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