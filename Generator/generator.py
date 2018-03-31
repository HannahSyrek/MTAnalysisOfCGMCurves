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
import sys  

'''
This class generates a train- and a testset to classify cgm curve progressions.
The Generator builds six different cgm patterns, a datamatrix with 50 
samples per categorie and writes them into a csv file.
'''
class Generator:
    
    #instances of the imported class to initialize the six different patterns
    #np.random.randint(low=200, high=260)
    C1 = Curve(1,175,85,100,23,180,70)  
    C2 = Curve(2,210,30,100,23,180,70)
    #C3 = Curve(3,280,105,100,250,180,70)    
    C4 = Curve(4,230,200,100,23,180,70)      
    #C5 = Curve(5,200,40,100,250,180,70)      
    C6 = Curve(6,230,115,210,23,180,70)
    global categories
    categories = [C1,C2,C4,C6]

    '''
    Method to build the datamatrix.
    There are 50 curve samples per categorie, every curve exists out of 60 values.
    '''  

    def builddataset():
        dataset = np.zeros((4,21))
        counter = 0
        for curve in categories:            
            if(counter < 4):
                for i in range(counter, counter +1):
                    for j in range(0,21):
                        #generate the specific curve
                        c = curve.generate()
                        #smooth curve with savitzky_golay filter
                        chat = savgol_filter(c, 5, 3)
                        while(len(chat) < 20):
                            chat = np.append(chat, 0)
                        chat = np.append(chat, curve.Categorie)
                        dataset[i][j] = chat[j]
                counter = counter + 1      
        return dataset
                       
        
    '''
    Method to save the data frame into a csv file.
    '''
    def saveData(datamatrix):
        df = pd.DataFrame(datamatrix)
        #save dataframe in train- respectively testset
        df.to_csv("train_set4.csv",  index=False)
        #df.to_csv("test_set.csv")
        

    #call needed methods to produce the dataset
    d = builddataset()
    saveData(d)

#=======================================================================================
# plot the means of the 50 samples per categorie to visualize their specific progression
#=======================================================================================
#    reload(sys)  
#    sys.setdefaultencoding('utf8')
#    c = np.zeros((4,20))
#    count = 0
#    while(count<4):
#        for j in range(0,99):
#            c[count][:] += d[j+(100*count)][:-1]
#        print c
#        c[count][:] /= 100
#        print c
#        count +=1
#
# 
#    t = np.asarray(range(0,20))
#    c1 = c[0][:]
#    c1hat = savgol_filter(c1, 5, 3)
#    c2 = c[1][:]
#    c2hat = savgol_filter(c2, 5, 3)
#    #c3 = c[2][:]
#    #c3hat = savgol_filter(c3, 51, 3)
#    c4 = c[2][:]
#    c4hat = savgol_filter(c4, 5, 3)
#    #c5 = c[4][:]
#    #c5hat = savgol_filter(c5, 51, 3)
#    c6 = c[3][:]
#    c6hat = savgol_filter(c6, 5, 3)                    
    
#    l1 = []
#    u1 = []
#
#    for i in range(0, 20):
#        l1 = np.append(l1, C1.Lowerthreshold)
#        u1 = np.append(u1, C1.Upperthreshold)
#    t_lu = np.asarray(range(0,20))
#    
#    curve1, = plt.plot(t,c1hat,label='Normal')
#    curve2, = plt.plot(t,c2hat,label='Bolus zu groß')
#    #curve3, = plt.plot(t,c3hat,label="3") 
#    curve4, = plt.plot(t,c4hat,label='Bolus zu klein')
#    #curve5, = plt.plot(t,c5hat,label="5")
#    curve6, = plt.plot(t,c6hat,label='Korrekturbolus')
#    upper, = plt.plot(t_lu, u1,'r--')
#    lower, = plt.plot(t_lu, l1,'r--')   
#    plt.legend(loc=1, fontsize = 'x-large' )
#    plt.axis([0, 20, 10, 400])
#    plt.ylabel('glucose content (mg/dL)')
#    plt.xlabel('timesteps')
#    plt.show()
    




