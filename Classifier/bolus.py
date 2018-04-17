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
cgm_data = skipmissingdata(cgmdata)
bolusdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = 0,
                         usecols = (9)) 
patterns = decode_classes(np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/categorized_all_3classes.csv",
                          delimiter = ",", dtype = None, skip_header = 1)) 
raw_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1)
alldata = np.vstack((cgmdata,bolusdata)).T 


#cgm_data = skipmissingdata(cgmdata)



                    
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
Method to write recognized patterns in raw data, to evaluate class quality.
''' 
def write_patterns(data, pats):
    #Todo: for mod data without night
    class_vector = ["" for x in range(len(data))]
    for ind, i in enumerate(pats):
        for jnd, j in enumerate(data):
            if(jnd<(len(data)-1) and data[jnd]==i[0] and data[jnd+1]==i[1] and data[jnd+2]==i[2]):
                # than: the  j-th pattern is the following time serie of length 20
                first_value = jnd
                if(i[-1]==1):
                    class_vector[first_value] = 'Class 1'
                elif(i[-1]==4):
                    class_vector[first_value]  = 'Class 4'
                elif(i[-1]==5):
                    class_vector[first_value]  = 'Class 5'
                elif(i[-1]==6):
                    class_vector[first_value]  = 'Class 6'

    # Return cgm values with associated class
    categorized_data = np.c_[np.asarray(data), class_vector] 
    _raw = np.asarray(pd.DataFrame(raw_data))
    df = pd.DataFrame(_raw)
    df.to_csv("Data/temporarystuff.csv", index=False)
    print _raw

    print categorized_data  
    #print categorized_data.T[0][0]
    #print categorized_data.T[1]
    count = 0
    print _raw[0][3]
    print categorized_data.T[0][0]
    for ind, i in enumerate(_raw):
        #print i[3]
        #print i.T[3][0]
        #print i[3],categorized_data.T[count][0]
        #rint categorized_data.T[count][0]
        #print count
        if (i[3]==categorized_data.T[count][0]):
            i[23] = categorized_data.T[count][1]
           #print i[23]
            count += 1
        else:
            ind += 1
            
    return _raw

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
class_data =  write_patterns(cgm_data,patterns)


df = pd.DataFrame(class_data)
df.to_csv("Data/bolus_3_classes_new_new.csv", index=False)
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
    