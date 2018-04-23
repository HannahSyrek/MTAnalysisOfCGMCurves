# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:36:23 2018

@author: hannah syrek
Script to label manually a test dataset to compute the accuracy rate of the implemented classifier.
"""
#Imports
import os
os.chdir("/home/hannah/Dokumente/MTAnalysisOfCGMCurves")
from utilities import *
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt



__data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v10.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1, usecols = [3]) 

#data_15 = []
#for count, i in enumerate(__data):
#    if(count % 3 ==0):
#        #print count
#        data_15.append(__data[count])
#        count +=1
#    else:
#        count +=1
#   
#data = np.array(skipmissingdata(data_15))
##print len(data)/20
#data.resize((len(data)/20,20))
#df = pd.DataFrame(data)
#df.to_csv("Classifier/Data/x.csv",  index=False) 


data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/labeled_v10.csv", 
                        delimiter = ",", dtype = None, skip_header = 1)

r= np.array(data)

k =540
j = k+1
l = k+2
i = k+3
m = k+4
n = k+5
o = k+6
print r[k]
print r[j]
print r[l]
print r[i]
print r[m]
print r[n]
print r[o]

plt.plot(r[k], label = k+2)
plt.plot(r[j], label = j+2)
plt.plot(r[l], label = l+2)
plt.plot(r[i], label = i+2)
plt.plot(r[m], label = m+2)
plt.plot(r[n], label = n+2)
plt.plot(r[o], label = o+2)
    


#_timeserie = np.zeros((len(realdata)-(ts_length-1),ts_length))
##save all time series with an overlap of 95 % in a new matrix 
#for i in range(0,len(realdata)-(ts_length-1)):
#    _timeserie[i][:] = realdata[i:ts_length+i]
#real_overlap = _timeserie 
#df = pd.DataFrame(real_overlap)
#df.to_csv("Classifier/Data/real_ModRaw_overlap.csv",  index=False)   
#print real_overlap 



#all_timeseries = np.zeros((9716,ts_length))
#for i in range(0,len(realdata)-(ts_length-1)):
#    all_timeseries[i][:] = realdata[i:ts_length+i]
#real_data = all_timeseries
#  
##df = pd.DataFrame(real_data)
##df.to_csv("Classifier/Data/datasetLabeled.csv",  index=False)   
#
#c_means=np.zeros((3,20))
#
#for i in range(0,20):
#    c_means[0][i] = np.mean(trainset.T[i][0:200])
#    c_means[1][i] = np.mean(trainset.T[i][200:400])
#    c_means[2][i] = np.mean(trainset.T[i][500:600])
    #c_means[3][i] = np.mean(trainset.T[i][600:800])

# c_means[0]
#c_means[0][-1] = 1.0 
#c_means[1][-1] = 2.0  
#c_means[2][-1] = 4.0  
#c_means[3][-1] = 6.0    
#df = pd.DataFrame(c_means)
#df.to_csv("Classifier/generalizedSamples/generalized_Curves.csv",  index=False)  
   
#tmpdata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/logdata.csv",
#                          delimiter = ",", dtype = None, skip_header = 1)
#for i in tmpdata:
#     plt.plot(i[-1],np.amax(i[:-1]), '*')
#
#reload(sys)  
#sys.setdefaultencoding('utf8')
#plt.axis([-1, 4, -10, 30])
#plt.ylabel('activationValue')
#plt.xlabel('classes')
#plt.show()
   
#class5_data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/class5_set.csv",
#                          delimiter = ",", dtype = None, skip_header = 1)
#for i in class5_data[0:50]:
#    print i[-1]
#    plt.plot(time_steps,i[:-1])
   


       
#print trainset[0][:-1]
#plt.plot(time_steps,c_means[0],label = "normal progression") 
#plt.plot(time_steps, c_means[1],label = "bolus to small")  
#plt.plot(time_steps, c_means[2],label = "correction of the bolus")  
#plt.plot(time_steps, c_means[3],label = "correction of the bolus") 


plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, 19, 10, 400])
plt.ylabel('blood glucose content (mg/dL)')
plt.xlabel('timesteps')
plt.show()

#k=154
#for i in range(k,k+1):
#    c=real_data[i][:]   
#    print real_data[i][:]
#    plt.plot(c,'m',label = i)












#==============================================================================
#plot the results to visualize the found patterns
#==============================================================================
#reload(sys)  
#sys.setdefaultencoding('utf8')
#plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
#plt.legend(loc=1)
#plt.axis([0, ts_length-1, 10, 400])
#plt.ylabel('blood glucose content (mg/dL)')
#plt.xlabel('timesteps')
#plt.show() 