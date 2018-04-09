# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:07:00 2018

@author: hannah syrek

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *



cgmdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))
bolusdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = 0,
                         usecols = (9)) 
patterns = decode_classes(np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/categorized_dataCNNnew.csv",
                          delimiter = ",", dtype = None, skip_header = 1)) 
alldata = np.vstack((cgmdata,bolusdata)).T 


print len(patterns[0])

                    
'''
Method to skip missing data and concatenate cgm and bolus values.
'''
def skipmissing_data(data):
    #Todo: zwei boli nacheinander fixen
    newcgmdata = []
    newbolusdata = []
    for count, i in enumerate(data):
        if(i[0] == -1 and i[1]==0):
            count += 1
        else:
            newcgmdata.append(i[0])
            newbolusdata.append(i[1]) 
    _temp = np.vstack((newcgmdata,newbolusdata)).T  
    for count, j in enumerate(_temp):
        if(j[0]== -1):
            _temp[count-1][1] = 1
            _temp[count][0] =  _temp[count+1][0] 
            _temp[count][1] =  _temp[count+1][1] 
            count += 1
        else:
            _temp[count][0] = j[0]
            _temp[count][1] = 0
    return _temp


'''
Method to write recognized patterns in raw data, to plot them with bolus values and evaluate class quality.
''' 
def write_patterns(data, pats):
    #Todo: for mod data without night
    for ind, i in enumerate(pats):
        for jnd, j in enumerate(data):
            if(jnd<(len(data)-1) and data[jnd][0]==i[0] and data[jnd+1][0]==i[1] and data[jnd+2][0]==i[2]):
                # than: the  jth pattern is the following 20 values
                # save class in data file
                # temp_ind = ind
                for temp in range(jnd, jnd+20):
                    #print temp
                    data[temp][2] = i[-1] 

    # Return cgm values with bolus and associated class
    return data


data = skipmissing_data(alldata) 
pattern_data = np.concatenate( (data,np.zeros((len(data),1)) ), axis = 1) 
class_data =  write_patterns(pattern_data,patterns)

df = pd.DataFrame(class_data)
df.to_csv("Data/bolus_class_dataset.csv", index=False)
data_transposed = class_data.T 
plt.plot(data_transposed[0])
plt.plot(data_transposed[1]*data_transposed[0], 'r*')
plt.plot(data_transposed[2]*10, 'g-')


# Initilize some parameters
low_value = 70
up_value = 180
lower_bound = []
upper_bound = []
# Fill time step vector to plot the cgm curvesover time
for i in range(0, len(data)):
    lower_bound.append(low_value)
    upper_bound.append(up_value)
time_steps = np.asarray(range(0,len(data)))   
#==============================================================================
#plot the results to visualize the found patterns
#==============================================================================
reload(sys)  
sys.setdefaultencoding('utf8')
plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, len(data)/25, 1, 500])
plt.ylabel('blood glucose content (mg/dL) with associated bolud value')
plt.xlabel('timesteps')
plt.show()    
    