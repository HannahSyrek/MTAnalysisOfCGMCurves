# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:35:00 2018

@author: Hannah syrek
This script set the characteristics of the curve instances and
implements the generation of curves.
"""

#Needed Imports
import numpy as np


class Curve(object):
    
    def __init__(self, categorie, maximum, minimum, normalvalue, time, upperthreshold, lowerthreshold): 
        self.Categorie = categorie        
        self.Maximum = maximum 
        self.Minimum = minimum 
        self.Normalvalue = normalvalue
        self.Time = time 
        self.Upperthreshold = upperthreshold
        self.Lowerthreshold = lowerthreshold
    
    
    def generate(self):
        curve = np.array([np.random.randint(low=self.Normalvalue-10, high=self.Normalvalue+10), 
                          np.random.randint(low=self.Normalvalue-10, high=self.Normalvalue+10)])
        firstpart = np.random.randint(low=self.Normalvalue, high=self.Maximum, size=(1,self.Time/4))
        sortedfirstpart = np.sort(firstpart)
        curve = np.append(curve, sortedfirstpart)
        curve = np.append(curve, np.random.randint(low=self.Maximum-5, high=self.Maximum+5))
        curve = np.append(curve, np.random.randint(low=self.Maximum-5, high=self.Maximum+5))
        curve = np.append(curve, np.random.randint(low=self.Maximum-5, high=self.Maximum+5))        
        secondpart = np.random.randint(low=self.Minimum, high=self.Maximum, size=(1,self.Time/4))
        sortedsecondpart = -np.sort(-secondpart)
        curve = np.append(curve, sortedsecondpart)
        if (self.Minimum < self.Lowerthreshold):
            thirdpart = np.random.randint(low=self.Minimum, high=self.Normalvalue, size=(1,self.Time/4))
            sortedthirdpart = np.sort(thirdpart)
            curve = np.append(curve, sortedthirdpart)
        else:   
            #print curve[-1]
            thirdpart = np.random.randint(low=curve[-1]-5, high=curve[-1]+5, size=(1,self.Time/4))
            #print thirdpart            
            curve = np.append(curve, thirdpart)
        return curve
      
    
        
        
        
  

