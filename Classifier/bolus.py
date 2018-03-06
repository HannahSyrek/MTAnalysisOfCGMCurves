# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 11:07:00 2018

@author: hannah syrek

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

cgmdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))
bolusdata = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = 0,
                         usecols = (9)) 
                         
alldata = np.vstack((cgmdata,bolusdata)).T 

                    


def skipmissingdata(data): 
    newcgmdata = []
    newbolusdata = []
    for count, i in enumerate(data):
        if(i[0] == -1):
            count += 1
        else:
            newcgmdata.append(i[0])
            newbolusdata.append(i[1])            
    return np.vstack((newcgmdata,newbolusdata)).T
    
    
data = skipmissingdata(alldata)  
df = pd.DataFrame(data)
df.to_csv("Data/bolusdataset.csv",  index=False)  
    
