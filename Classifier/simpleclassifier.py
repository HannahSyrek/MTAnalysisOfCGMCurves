# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018

@author: hannah syrek
This script implements the DTW algorithm to measure the distances between time
series and classifies them with k-means.
"""

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Simpleclassifier():

    data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", delimiter = ",", dtype = None) 

    curve1 = data[0]
    curve11 = data[1]
    curve2 = data[50]
    curve22 = data[51]
    curve3 = data[100]
    curve4 = data[150]
    curve5 = data[200]
    curve6 = data[250]
    curve33 = data[101]
    curve44 = data[151]
    curve55 = data[201]
    curve66 = data[251]
    print curve3

    #good size for w seems to be w=100
    def DTWDistanceFast(s1, s2,w):
        DTW={}
        w = max(w, abs(len(s1)-len(s2)))
        for i in range(-1,len(s1)):
            for j in range(-1,len(s2)):
                DTW[(i, j)] = float('inf')
        DTW[(-1, -1)] = 0
        for i in range(len(s1)):
            for j in range(max(0, i-w), min(len(s2), i+w)):
                dist= (s1[i]-s2[j])**2
                DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
        return np.sqrt(DTW[len(s1)-1, len(s2)-1])
    
    

    #Use LBKeogh lower bund of dynamic time wraping to speed up the computation
    # good size for r seems to be r=10
    def LB_Keogh(s1,s2,r):
        LB_sum=0
        for ind,i in enumerate(s1):
            lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
            if i>upper_bound:
                LB_sum=LB_sum+(i-upper_bound)**2
            elif i<lower_bound:
                LB_sum=LB_sum+(i-lower_bound)**2
        return np.sqrt(LB_sum)



    def knn(train,test,w):
        preds=[]
        for ind,i in enumerate(test):
            min_dist=float('inf')
            closest_seq=[]
            #print ind
            for j in train:
                if Simpleclassifier.LB_Keogh(i[:-1],j[:-1],10)<min_dist:
                    dist=Simpleclassifier.DTWDistanceFast(i[:-1],j[:-1],w)
                    if dist<min_dist:
                        min_dist=dist
                        closest_seq=j
            preds.append(closest_seq[-1])
        return classification_report(test[:,-1],preds)

    
#==============================================================================
#call some methodes and prints to test stuff
#==============================================================================
    print ("Distanz 1 zur 1 Kurve:") , DTWDistanceFast(curve1, curve11, 100)
    print ("Distanz 1 zur 2 Kurve:") , DTWDistanceFast(curve1, curve2, 100)
    print ("Distanz 1 zur 3 Kurve:") , DTWDistanceFast(curve1, curve3, 100)
    print ("Distanz 1 zur 4 Kurve:") , DTWDistanceFast(curve1, curve4, 100)
    print ("Distanz 1 zur 5 Kurve:") , DTWDistanceFast(curve1, curve5, 100)
    print ("Distanz 1 zur 6 Kurve:") , DTWDistanceFast(curve1, curve6, 100)
    print(" ")   
    print ("Distanz 2 zur 2 Kurve:") , DTWDistanceFast(curve2, curve22, 100)
    print ("Distanz 2 zur 3 Kurve:") , DTWDistanceFast(curve2, curve3, 100)
    print ("Distanz 2 zur 4 Kurve:") , DTWDistanceFast(curve2, curve4, 100)
    print ("Distanz 2 zur 5 Kurve:") , DTWDistanceFast(curve2, curve5, 100)
    print ("Distanz 2 zur 6 Kurve:") , DTWDistanceFast(curve2, curve6, 100)
    print(" ")   
    print ("Distanz 3 zur 3 Kurve:") , DTWDistanceFast(curve3, curve33, 100)    
    print ("Distanz 3 zur 4 Kurve:") , DTWDistanceFast(curve3, curve4, 100)
    print ("Distanz 3 zur 5 Kurve:") , DTWDistanceFast(curve3, curve5, 100)
    print ("Distanz 3 zur 5 Kurve:") , DTWDistanceFast(curve3, curve6, 100)
    print(" ")   
    print ("Distanz 4 zur 4 Kurve:") , DTWDistanceFast(curve4, curve44, 100)    
    print ("Distanz 4 zur 5 Kurve:") , DTWDistanceFast(curve4, curve5, 100)
    print ("Distanz 4 zur 6 Kurve:") , DTWDistanceFast(curve4, curve6, 100)
    print(" ")   
    print ("Distanz 5 zur 5 Kurve:") , DTWDistanceFast(curve5, curve55, 100)
    print ("Distanz 5 zur 6 Kurve:") , DTWDistanceFast(curve5, curve6, 100)
    print(" ")          
    print ("Distanz 6 zur 6 Kurve:") , DTWDistanceFast(curve6, curve66, 100)
    print("")
    print("")
        
    print ("Distanz 1 zur 1 Kurve:") , LB_Keogh(curve1, curve11, 10)
    print ("Distanz 1 zur 2 Kurve:") , LB_Keogh(curve1, curve2, 10)
    print ("Distanz 1 zur 3 Kurve:") , LB_Keogh(curve1, curve3, 10)
    print ("Distanz 1 zur 4 Kurve:") , LB_Keogh(curve1, curve4, 10)
    print ("Distanz 1 zur 5 Kurve:") , LB_Keogh(curve1, curve5, 10)
    print ("Distanz 1 zur 6 Kurve:") , LB_Keogh(curve1, curve6, 10)
    print(" ")   
    print ("Distanz 2 zur 2 Kurve:") , LB_Keogh(curve2, curve22, 10)
    print ("Distanz 2 zur 3 Kurve:") , LB_Keogh(curve2, curve3, 10)
    print ("Distanz 2 zur 4 Kurve:") , LB_Keogh(curve2, curve4, 10)
    print ("Distanz 2 zur 5 Kurve:") , LB_Keogh(curve2, curve5, 10)
    print ("Distanz 2 zur 6 Kurve:") , LB_Keogh(curve2, curve6, 10)
    print(" ")   
    print ("Distanz 3 zur 3 Kurve:") , LB_Keogh(curve3, curve33, 10)    
    print ("Distanz 3 zur 4 Kurve:") , LB_Keogh(curve3, curve4, 10)
    print ("Distanz 3 zur 5 Kurve:") , LB_Keogh(curve3, curve5, 10)
    print ("Distanz 3 zur 5 Kurve:") , LB_Keogh(curve3, curve6, 10)
    print(" ")   
    print ("Distanz 4 zur 4 Kurve:") , LB_Keogh(curve4, curve44, 10)    
    print ("Distanz 4 zur 5 Kurve:") , LB_Keogh(curve4, curve5, 10)
    print ("Distanz 4 zur 6 Kurve:") , LB_Keogh(curve4, curve6, 10)
    print(" ")   
    print ("Distanz 5 zur 5 Kurve:") , LB_Keogh(curve5, curve55, 10)
    print ("Distanz 5 zur 6 Kurve:") , LB_Keogh(curve5, curve6, 10)
    print(" ")          
    print ("Distanz 6 zur 6 Kurve:") , LB_Keogh(curve6, curve66, 10)

    
    
    
    
    
    
    
    
    
