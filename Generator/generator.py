# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:30:00 2018

@author: Hannah syrek
This is a script to generate data to train the classifier for the categorization of CGM curves.
"""

#Needed Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from curve import Curve
import sys  


'''
This class generates a train- and a testset to classify cgm curve progressions.
The Generator builds six different cgm patterns, a datamatrix with 50 
samples per categorie and writes them into a csv file.
'''
class Generator:
    
    #instances of the imported class to initialize the six different patterns
    #np.random.randint(low=200, high=260)
    C1 = Curve(1,180,70,80,23,180,70)  
    #C2 = Curve(0,200,30,100,23,180,70)
    #C3 = Curve(3,280,105,100,250,180,70)    
    C4 = Curve(4,230,200,90,23,180,70)  
    # Residue class    
    C51 = Curve(5,110,90,100,23,180,70)  
    C52 = Curve(5,300,10,100,23,180,70)
    C53 = Curve(5,105,10,100,23,180,70)  
    C54 = Curve(5,350,100,20,23,180,70) 
    
    C6 = Curve(6,260,80,250,23,180,70)
    global categories
    categories = [C1,C4,C6]



    '''
    Method to build the datamatrix.
    There are 50 curve samples per categorie, every curve exists out of 60 values.
    '''  

    def builddataset():
        dataset = np.zeros((60,21))
        counter = 0
        for curve in categories:            
            if(counter < 60):
                for i in range(counter, counter +20):
                    #generate the specific curve
                    c = curve.generate()
                    #smooth curve with savitzky_golay filter
                    chat = savgol_filter(c, 5, 3)
                    chat = np.append(chat, curve.Categorie)
                    for j in range(0,21):
                        dataset[i][j] = chat[j]
                counter = counter + 20    
        return dataset
                       
        
    '''
    Method to save the data frame into a csv file.
    '''
    def saveData(datamatrix):
        df = pd.DataFrame(datamatrix)
        #save dataframe in train- respectively testset
        df.to_csv("test.csv",  index=False)
        #df.to_csv("test_set.csv")
        

    #call needed methods to produce the dataset
    d = builddataset()
    saveData(d)

#=======================================================================================
# plot the means of the 50 samples per categorie to visualize their specific progression
#=======================================================================================
    reload(sys)  
    sys.setdefaultencoding('utf8')  
    plt.legend(loc=1, fontsize = 'x-large' )
    plt.axis([0, 20, 10, 400])
    plt.ylabel('glucose content (mg/dL)')
    plt.xlabel('timesteps')
    plt.show()
    




