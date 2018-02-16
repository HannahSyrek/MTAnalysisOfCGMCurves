# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:30:00 2018

@author: hannah syrek
This is a script to generate data to train the classifier for the categorization of CGM curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import factorial
from scipy.interpolate import spline
from curve import Curve


'''
This class generates a train- and a testset to classify cgm curve progressions.
The Generator builds six different cgm patterns, a datamatrix with 50 
samples per categorie and writes them into a csv file.
'''
class Generator:
    
    #instances of the imported class to initialize the six different patterns
    #For time duration of 3-5 hours: np.random.randint(low=50, high=70)
    C1 = Curve(1,170,85,100,np.random.randint(low=200, high=260),180,70)  
    C2 = Curve(2,170,50,100,np.random.randint(low=200, high=260),180,70)
    C3 = Curve(3,250,105,100,np.random.randint(low=200, high=260),180,70)    
    C4 = Curve(4,220,200,100,np.random.randint(low=200, high=260),180,70)      
    C5 = Curve(5,200,40,100,np.random.randint(low=200, high=260),180,70)      
    C6 = Curve(6,230,115,210,np.random.randint(low=200, high=260),180,70)

    global categories
    categories = [C1,C2,C3,C4,C5,C6]

    '''
    Method to build the datamatrix.
    There are 50 curve samples per categorie, every curve exists out of 60 values.
    '''    
    def builddataset():
        dataset = np.zeros((300,252))
        counter = 0
        for curve in categories:            
            if(counter < 300):
                for i in range(counter, counter +50):
                    for j in range(0,252):
                        c = curve.generate()
                        while(len(c) < 251):
                            c = np.append(c, 0)  
                        c = np.append(c, curve.Categorie)
                        dataset[i][j] = c[j]
                counter = counter + 50      
        return dataset

        
    '''
    Method to save the data frame into a csv file.
    '''
    def saveData(datamatrix):
        df = pd.DataFrame(datamatrix)
        df.to_csv("trainset.csv")

     
    #call needed methods to produce the dataset
    #d = builddataset()
    #saveData(d)
 


    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        """
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """   
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
                        

#==============================================================================
# plot one sample out of each categorie to visualize their specific progression
#==============================================================================
 
    t1 = np.asarray(range(0,len(C1.generate())))
    c1 = C1.generate()
    c1hat = savitzky_golay(c1, 51, 3)
    t2 = np.asarray(range(0,len(C2.generate())))
    c2 = C2.generate()
    c2hat = savitzky_golay(c2, 51, 3)
    t3 = np.asarray(range(0,len(C3.generate())))
    c3 = C3.generate()
    c3hat = savitzky_golay(c3, 51, 3)
    t4 = np.asarray(range(0,len(C4.generate())))
    c4 = C4.generate()
    c4hat = savitzky_golay(c4, 51, 3)
    t5 = np.asarray(range(0,len(C5.generate())))
    c5 = C5.generate()
    c5hat = savitzky_golay(c5, 51, 3)
    t6 = np.asarray(range(0,len(C6.generate()))) 
    c6 = C6.generate()
    c6hat = savitzky_golay(c6, 51, 3)
    maxcurve = max(len(c1),len(c2),len(c3),len(c4),len(c5),len(c6))
    l1 = []
    u1 = []
    for i in range(0, maxcurve):
        l1 = np.append(l1, C1.Lowerthreshold)
        u1 = np.append(u1, C1.Upperthreshold)
    t_lu = np.asarray(range(0,maxcurve))
    
    plt.plot(t1,c1hat, t2,c2hat, t3,c3hat, t4,c4hat, t5,c5hat, t6,c6hat, t_lu, u1,'r--' , t_lu, l1, 'r--')
    plt.axis([0, maxcurve, 10, 400])
    plt.ylabel('glucose content (mg/dL)')
    plt.xlabel('timesteps')
    plt.show()
    




