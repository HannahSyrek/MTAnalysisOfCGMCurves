# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:07:00 2018
Script to export the recognized patterns into raw data.
@author: Hannah syrek

"""

# Needed Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from utilities import *


cgmdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))
cgm_data = skipmissingdata(cgmdata)
patterns = decode_classes(np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/CNN_labeled.csv",
                          delimiter = ",", dtype = None, skip_header = 1))
temp_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1)
raw_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 0)



'''
Method to write the recognized patterns in raw data set, to evaluate class quality.
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
df.to_csv("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Final_Classifications/CNN/testClassifications.csv", index=False, header = False)
reload(sys)  
sys.setdefaultencoding('utf8')
plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, len(data)/40, 1, 500])
plt.ylabel('blood glucose content (mg/dL) with associated bolus value')
plt.xlabel('timesteps')
plt.show()    
    
