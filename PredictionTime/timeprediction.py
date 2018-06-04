# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:14:23 2018
@author: hannah syrek
"""
# Imports
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Read data
time_data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/PredictionTime/predictiontime.csv", 
                         delimiter = ",", dtype = None, skip_header = 1, usecols = [1,2,3,4,5])

print time_data[1]
#print time_data.T


time_vector = [50,100,150,200,250]

print time_vector
                         
# Plot data
reload(sys)  
sys.setdefaultencoding('utf8')

plt.plot(time_vector, time_data[0], 'ro-', label = "VBDTW")
plt.plot(time_vector, time_data[1], 'g*-', label = "DDTW")
plt.plot(time_vector, time_data[2]/5, 'b^-', label = "FBDTW")
plt.plot(time_vector, time_data[3]/5, 'yd-', label = "AFBDTW")
plt.plot(time_vector, time_data[4], 'mv-', label = "CNN")

plt.legend(loc=2)
plt.axis([40,260, 0, 500])
plt.ylabel('Prediction Time (sec)')
plt.xlabel('Size of Training Set')
plt.grid(True)
plt.show()                         