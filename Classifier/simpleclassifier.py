# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 09:15:20 2018
@author: hannah syrek
This script implements the DTW algorithm to measure the distances between time
series, a simple knn-classification algorithm and k-means to find patterns in
the datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


#==============================================================================
#read data from .csv files
#==============================================================================
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1) 
testset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/testset.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
realdata1 = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3)) 
realdata2 = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v10.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1,
                         usecols = (3))  
tempcatdata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/catdataset.csv",
                          delimiter = ",", dtype = None, skip_header = 1) 

'''
Implements a faster version of dynamic time wraping, includes w, a windows size. 
Only mappings within this window size are considered which speeds up the inner
loop of the computations. Good size for w seems to be w = 50.
'''
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
 
    
'''
Implements LB_Keogh lower bound of dynamic timewraping to speed up the computations.
Good size for r seems to be r=10.
'''
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


'''
Implements the k-NearestNeighbor classification to determine the similarity 
between the time series and classify the dataset with real cgm values.
(Takes round about 5 minutes to run.)
'''
def knnStatic(train,test,w):
    predictions=[]
    #categorize die time series
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        print ind 
        for j in train:
            if LB_Keogh(i[:],j[:-1],10)<min_dist:
                dist=DTWDistanceFast(i[:],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist     
                    closest_seq=j
        predictions.append(closest_seq[-1])
    #produce the categorized dataset: catdata
    predsT = np.array([predictions]).T
    realdata = np.array(test)
    catdata = np.concatenate((realdata, predsT), axis = 1)  
    df = pd.DataFrame(catdata)
    df.to_csv("Data/catdataset.csv",  index=False)     
    category = 1
    meandata = np.zeros((4,20))
    #average the curves of the particular category
    while(category<7):
        for j in range(0,4):
            count = 0
            for i in range(0,len(predictions)):      
                if(catdata[i][-1]==category):
                    meandata[j][:] += catdata[i][:-1]
                    count += 1           
            meandata[j][:] /= count
            category += 1
            if (category==3 or category==5):
                category += 1
    #return the averaged curves, one per categorie         
    return meandata


'''
Implements the k-NearestNeighbor classification to determine the similarity 
between time series, iterate point by point over the hole dataset to find the 
progression of the categories.
(Takes round about 90 minutes for 9716 time series to run.)
'''
def knnDynamic(train,test,w):
    predictions=[]
    dyn_timeserie = np.zeros((9716,20))
    #save all possible 9716 time series in a new matrix to iterate over all 
    for i in range(0,len(test)-19):
        dyn_timeserie[i][:] = test[i:20+i]
    test = dyn_timeserie   
    #categorize all time series 
    for ind,i in enumerate(test):        
        min_dist=float('inf')
        closest_seq=[]
        print ind
        for j in train:
            if LB_Keogh(i[:],j[:-1],10)<min_dist:
                dist=DTWDistanceFast(i[:],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist     
                    closest_seq=j
        #assign all time series with a higher distance as 70 to a separate catgeory
        if(min_dist>70):
            predictions.append(5.0)
        else:
            predictions.append(closest_seq[-1])                     
    #produce the categorized dataset: catdata
    #attention: the data includes repetitions of the assigned curves-> skipRepetitions
    predsT = np.array([predictions]).T
    realdata = np.array(test)
    catdata = np.concatenate((realdata, predsT), axis = 1)  
    df = pd.DataFrame(catdata)
    df.to_csv("Data/catdataset.csv",  index=False)     

    return catdata



'''
Skip the missing cgm values in the real data. Missing values were previously 
replaced by -1.
'''
def skipmissingdata(data): 
    newdata = []
    for count, i in enumerate(data):
        if(i == -1):
            count += 1
        else:
            newdata.append(i)            
    return newdata
   
 
'''
Method to skip the repetitions in the dynamic categorized data.
'''
def skipRepetitions(data):    
    catdata_dyn = np.zeros((len(data),21))
    catdata_dyn[0][:] = tempcatdata[0][:]
    count = 0
    ind = 1
    while(ind < len(data)):
        if(catdata_dyn[count][-1] != data[ind][-1]):  
            catdata_dyn[count+1][:] = data[ind][:]
            count +=1
            ind +=1
        else:
            ind +=1
            
    df = pd.DataFrame(catdata_dyn)
    df.to_csv("Data/catdataWithoutRepetitions.csv",  index=False)       
    return catdata_dyn



 
'''
Method to plot all assigned curves of the particular categorie.
'''        
def plotCategories(category): 
    data = skipRepetitions(tempcatdata)     
    count = 0
    for i in data:
        if(i[-1]==category):
            plt.plot(i)
            count += 1
    return count        
 

#==============================================================================
#plot the results to visualize the found patterns
#==============================================================================
reload(sys)  
sys.setdefaultencoding('utf8')
Data = skipmissingdata(realdata1)
realdata = np.array(Data)
#uncomment the resize method for knnStatic and averaged dtw curves
#realdata.resize(486,20) 
#curves = knnStatic(trainset,realdata,50)
#print knnDynamic(trainset,realdata, 50)
print plotCategories(6)
                        
#fill vector u1 and l1 with the upper- and lower boundvalue to draw them into 
#the visualization
l1 = []
u1 = []
for i in range(0, 20):
    l1 = np.append(l1, 70)
    u1 = np.append(u1, 180)
t_lu = np.asarray(range(0,20))
plt.plot(t_lu, u1,'r--', t_lu, l1, 'r--')
plt.legend(loc=1)
plt.axis([0, 19, 10, 400])
plt.ylabel('glucose content (mg/dL)')
plt.xlabel('timesteps: one measurement every 15 minutes')
plt.show()






#==============================================================================
# Stuff, der noch gebraucht werden könnte
#==============================================================================

##################_mean of the dtw data, seems to be not the best idea###
#    category = 1
#    meandata = np.zeros((5,20))
#    #average the curves of the particular category
#    while(category<7):
#        for j in range(0,5):
#            count = 0
#            for i in range(0,len(predictions)):      
#                if(catdata[i][-1]==category):
#                    meandata[j][:] += catdata[i][:-1]
#                    count += 1           
#            meandata[j][:] /= count
#            category += 1
#            if (category==3):
#                category += 1
    #return the averaged curves, one per categorie 


#curves =  knnDynamic(trainset,realdata, 50)
#plt.plot(curves[0], label= '1: Normal')
#plt.plot(curves[1], label= '2: Bolus zu groß')
#plt.plot(curves[2], label= '4: Bolus zu klein')
#plt.plot(curves[3], label= '5: Keine passende Kategorie')
#plt.plot(curves[4], label= '6: Korrekturbolus')

#df = pd.DataFrame(realdataset)
#df.to_csv("realdataset.csv",  index=False)

#stack the train- and testset together to have a bigger dataset
#entiredataset = np.vstack((trainset[:,:-1],testset[:,:-1]))

#plot the final centroids of the particular cluster 
#centroids = k_means_clust(realdataset,6,15,100)
#for i in centroids:   
    #plt.plot(i)


#D2 = skipmissingdata(realdata1)
#D2new = np.array(D2) 
#D2new.resize(486,20)
#stack the two realsets together to have a bigger dataset
#realdataset = np.vstack((D1new[:,:],D2new[:,:]))



    
    
    
    
    
    
    
    
    
