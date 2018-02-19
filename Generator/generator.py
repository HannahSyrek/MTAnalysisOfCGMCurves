# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:30:00 2018

@author: hannah syrek
This is a script to generate data to train the classifier for the categorization of CGM curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from curve import Curve

'''
This class generates a train- and a testset to classify cgm curve progressions.
The Generator builds six different cgm patterns, a datamatrix with 50 
samples per categorie and writes them into a csv file.
'''
class Generator:
    
    #instances of the imported class to initialize the six different patterns
    #np.random.randint(low=200, high=260)
    C1 = Curve(1,170,85,100,250,180,70)  
    C2 = Curve(2,185,50,100,250,180,70)
    C3 = Curve(3,280,105,100,250,180,70)    
    C4 = Curve(4,220,200,100,250,180,70)      
    C5 = Curve(5,200,40,100,250,180,70)      
    C6 = Curve(6,230,115,210,250,180,70)
    global categories
    categories = [C1,C2,C3,C4,C5,C6]

    '''
    Method to build the datamatrix.
    There are 50 curve samples per categorie, every curve exists out of 60 values.
    '''    
    def builddataset():
        dataset = np.zeros((300,242))
        counter = 0
        for curve in categories:            
            if(counter < 300):
                for i in range(counter, counter +50):
                    for j in range(0,242):
                        #generate the specific curve
                        c = curve.generate()
                        #smooth curve with savitzky_golay filter
                        chat = savgol_filter(c, 51, 3)
                        while(len(chat) < 241):
                            chat = np.append(chat, 0)
                        chat = np.append(chat, curve.Categorie)
                        dataset[i][j] = chat[j]
                counter = counter + 50      
        return dataset
                       
        
    '''
    Method to save the data frame into a csv file.
    '''
    def saveData(datamatrix):
        df = pd.DataFrame(datamatrix)
        #save dataframe in train- respectively testset
        df.to_csv("testset.csv")
        #df.to_csv("testset.csv")
        

    #call needed methods to produce the dataset
    d = builddataset()
    saveData(d)

#==============================================================================
# plot one sample out of each categorie to visualize their specific progression
#==============================================================================
 
    t1 = np.asarray(range(0,len(C1.generate())))
    c1 = C1.generate()
    c1hat2 = savgol_filter(c1, 51 ,3)
    t2 = np.asarray(range(0,len(C2.generate())))
    c2 = C2.generate()
    c2hat = savgol_filter(c2, 51, 3)
    t3 = np.asarray(range(0,len(C3.generate())))
    c3 = C3.generate()
    c3hat = savgol_filter(c3, 51, 3)
    t4 = np.asarray(range(0,len(C4.generate())))
    c4 = C4.generate()
    c4hat = savgol_filter(c4, 51, 3)
    t5 = np.asarray(range(0,len(C5.generate())))
    c5 = C5.generate()
    c5hat = savgol_filter(c5, 51, 3)
    t6 = np.asarray(range(0,len(C6.generate()))) 
    c6 = C6.generate()
    c6hat = savgol_filter(c6, 51, 3)
    maxcurve = max(len(c1),len(c2),len(c3),len(c4),len(c5),len(c6))
    l1 = []
    u1 = []
    for i in range(0, maxcurve):
        l1 = np.append(l1, C1.Lowerthreshold)
        u1 = np.append(u1, C1.Upperthreshold)
    t_lu = np.asarray(range(0,maxcurve))
    
    plt.plot(t1,c1hat2, t2,c2hat, t3,c3hat, t4,c4hat, t5,c5hat, t6,c6hat, t_lu, u1,'r--' , t_lu, l1, 'r--')
    plt.axis([0, maxcurve, 10, 400])
    plt.ylabel('glucose content (mg/dL)')
    plt.xlabel('timesteps')
    plt.show()
    




