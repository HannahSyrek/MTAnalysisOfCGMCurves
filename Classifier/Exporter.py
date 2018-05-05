# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:07:00 2018
Script to export the recognized patterns into raw data.
@author: Hannah syrek

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *


cgmdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))
cgm_data = skipmissingdata(cgmdata)
patterns = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/withoutReps.csv",
                          delimiter = ",", dtype = None, skip_header = 1)
temp_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1)
raw_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 0)



'''
Method to write recognized patterns in raw data, to evaluate class quality.
''' 
def export(data, pats):
    class_vector = ["" for x in range(len(data))]
    for ind, i in enumerate(pats):
        for jnd, j in enumerate(data):
            if(jnd<(len(data)-1) and data[jnd]==i[0] and data[jnd+1]==i[1] and data[jnd+2]==i[2]):
                # than: the  j-th pattern is the following time serie of length 20
                first_value = jnd
                if(i[-1]==1):
                    class_vector[first_value] = 'BOLUS_CLASS1'
                elif(i[-1]==4):
                    class_vector[first_value]  = 'BOLUS_CLASS4'
                elif(i[-1]==5):
                    class_vector[first_value]  = 'BOLUS_CLASS5'
                elif(i[-1]==6):
                    class_vector[first_value]  = 'BOLUS_CLASS6'
    # Cgm values with associated class
    categorized_data = np.c_[np.asarray(data), class_vector] 
    count = 0
    # Write the classes into the raw data file
    for ind, i in enumerate(temp_data):
        if (count<len(categorized_data)-1 and float(i[3])==float(categorized_data.T[0][count])):
            raw_data[ind+1][25] = categorized_data.T[1][count]
            count += 1
        else:
            ind += 1
    return raw_data


# Save raw data with new information: The classification of the particular time serie
df = pd.DataFrame(export(cgm_data,patterns))
df.to_csv("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Final_Classifications/VBDTW/classified_data.csv", index=False, header = False)








#==============================================================================
# Stuff, could still be useful in future tasks
#==============================================================================

#bolusdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
#                         delimiter = ",", dtype = None, skip_header = 1, filling_values = 0,
#                         usecols = (9)) 
#alldata = np.vstack((cgmdata,bolusdata)).T 
                   
#'''
#Method to skip missing data and concatenate cgm and bolus values.
#'''
#def skipmissing_data(data):
#    #Todo: zwei boli nacheinander fixen
#    newcgmdata = []
#    newbolusdata = []
#    for count, i in enumerate(data):
#        if(i[0] == -1 and i[1]==0):
#            count += 1
#        else:
#            newcgmdata.append(i[0])
#            newbolusdata.append(i[1]) 
#    _temp = np.vstack((newcgmdata,newbolusdata)).T  
#    for count, j in enumerate(_temp):
#        if(j[0]== -1):
#            _temp[count-1][1] = 1
#            _temp[count][0] =  _temp[count+1][0] 
#            _temp[count][1] =  _temp[count+1][1] 
#            count += 1
#        else:
#            _temp[count][0] = j[0]
#            _temp[count][1] = 0
#    return _temp







#np.asarray(raw_data)
#d = np.asmatrix(pd.DataFrame(raw_data))
#print pd.DataFrame(raw_data)
#print d.T[3]

#'''
#Method to export recognized patterns in raw data.
#'''
#def exporter(cat_data, raw_data):
#    d = np.asmatrix(pd.DataFrame(raw_data)).T
#    # index = 3 -> cgm values, index = 23 -> class info
#    #for 
#
#    
#    return stuff
    


#data = skipmissing_data(alldata) 
#pattern_data = np.concatenate( (data,np.zeros((len(data),5)) ), axis = 1)


#data_transposed = class_data.T 
##plt.plot(data_transposed[0])
##plt.plot(data_transposed[1]*data_transposed[0], 'r*')
##plt.plot(data_transposed[2]*200, label='class 1')
##plt.plot(data_transposed[4]*50, label='class 4')
##plt.plot(data_transposed[6]*33, label='class 6')
#
#
#
#  
##==============================================================================
##plot the results to visualize the found patterns
##==============================================================================
## Initilize some parameters
#low_value = 70
#up_value = 180
#lower_bound = []
#upper_bound = []
## Fill time step vector to plot the cgm curvesover time
#for i in range(0, len(data)):
#    lower_bound.append(low_value)
#    upper_bound.append(up_value)
#time_steps = np.asarray(range(0,len(data))) 
#
#reload(sys)  
#sys.setdefaultencoding('utf8')
#plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
#plt.legend(loc=1)
#plt.axis([0, len(data)/40, 1, 500])
#plt.ylabel('blood glucose content (mg/dL) with associated bolus value')
#plt.xlabel('timesteps')
#plt.show()    
    