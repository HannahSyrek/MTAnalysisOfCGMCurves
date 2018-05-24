# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:06:59 2018
@author: hannah syrek
This script implements some needed utilities and parameter to run the classification 
algorithms.
"""

# Needed imports
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Initilize some parameters
ts_length = 20
low_value = 70
up_value = 180
lower_bound = []
upper_bound = []
global w_i
global w_j

# Fill time step vector to plot the cgm curves over time
for i in range(0, ts_length):
    lower_bound.append(low_value)
    upper_bound.append(up_value)
time_steps = np.asarray(range(0,ts_length))


# Read the needed data from .csv files
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/train_generated.csv", 
                         delimiter = ",", dtype = None, skip_header = 1)
labeled_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/testset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1)                         
_data = np.genfromtxt("/home/hannah/Dokumente/TSAd1/Datasets/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1, usecols = [3])


ddtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/DDTW_labeled_trainset.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
vbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/VBDTW_labeled_test.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
afbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/AFBDTW_labeled_trainset.csv", 
                        delimiter = ",", dtype = None, skip_header = 1)
fbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/fbdtw_labeled_traingen.csv", 
                        delimiter = ",", dtype = None, skip_header = 1)
cnndata = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/CNN_labeled.csv",
                          delimiter = ",", dtype = None, skip_header = 1)                          
x_data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/VBDTW_labeled_raw.csv", 
                        delimiter = ",", dtype = None, skip_header = 1)

                     

'''
Method to modify the raw data, skip all cgm values between 0 and 6 o'clock.
'''
def modify_rawData(data):
    mod_data = []
    for ind, i in enumerate(data):
        (h,m) = i[0].split(':')
        _hours = int(h)
        # Skip night values
        if(0<=_hours<=5):
            ind +=1
        else:
            mod_data.append(i)
            ind +=1            
    raw_data =[]    
    for ind,i in enumerate(mod_data):
        # Skip missing data
        if(i[1]==-1):
            ind+=1
        else:
            raw_data.append(i[1])
            ind+=1
    return raw_data
       
                          
'''
Method to decode class labels of the classified data,its a special method for 
the CNN data, caused by the encoding in the calculations.
'''
def decode_classes(data):    
    for i in data:
        if(i[-1]==0 and i[0]!=0):
            i[-1] = 1
        elif(i[-1]==1):
            i[-1] = 4
        elif(i[-1]==2):
            i[-1] = 6
        elif(i[-1]==3):
            i[-1] = 5
    return data    

cnn_data = decode_classes(cnndata) 
#cnn_data.resize((786,21))
#df = pd.DataFrame(cnn_data)
#df.to_csv("Data/richtigeLabel.csv", index=False) 


'''
Skip the missing cgm values in the real data. Missing values were previously 
replaced by -1.
'''
def skipmissingdata(data): 
    new_data = []
    for count, i in enumerate(data):
        if(i==-1):
            count+=1
        else:
            new_data.append(i)            
    return new_data
 
raw_data = skipmissingdata(_data) 

'''
Method to skip the repetitions in the dynamic categorized data.
'''
def skipRepetitions(data):    
    cat_data = np.zeros((len(data),ts_length+1))
    cat_data[0][:] = data[0][:-2]
    count = 0
    ind = 1
    while(ind<len(data)-2):
        if(cat_data[count][-1] != data[ind][-3]):  
            if(data[ind][-3] == data[ind+1][-3]):
                tmp_ind = ind
                dists = []
                while(ind<len(data)-1 and data[ind][-3]==data[ind+1][-3]):
                    dists = np.append(dists, data[ind][-2])
                    ind +=1
                min_loc = np.argmin(dists)
                cat_data[count+1][:] = data[(tmp_ind + min_loc)][:-2]
                count +=1
            else:
                cat_data[count+1][:] = data[ind][:-2]
                count +=1
                ind +=1
        else:
            ind +=1        
    df = pd.DataFrame(cat_data)
    df.to_csv("Data/withoutReps.csv", index=False)       
    return cat_data

 
'''
Method to plot all assigned curves of the particular category.
'''        
def plotCategories(category, data): 
    count = 0
    for i in data:
        if(i[-1]==category):
            plt.plot(i)
            count += 1
    return count        


'''
Method to calculate the derivation of a given point, as it is used in
[Keogh, E. J., & Pazzani, M. J. (2001, April). Derivative dynamic time warping.
In Proceedings of the 2001 SIAM International Conference on Data Mining
(pp. 1-11). Society for Industrial and Applied Mathematics]. Except the first 
and the last value of the timeserie. Its derivation is computed dependend only
on the second respectively the penultimate value. 
'''
def derive(ts):
    new_timeserie = np.zeros((len(ts))) 
    #set the first and the last derivation each depending on the follwoing respectively the previous point
    new_timeserie[0] = (ts[1]-ts[0]) 
    new_timeserie[-1] = (ts[-1]-ts[-2])
    #compute the derivation of every point in the time serie
    for i in range(1, len(ts)-1):
        new_timeserie[i] = float((ts[i]-ts[i-1]) + ((ts[i+1]-ts[i-1])/2))/2        
    return new_timeserie


'''
Method to compute the local feature of a datapoint in the given time serie, as it is introduced in 
[Xie, Y., & Wiltgen, B. (2010). Adaptive feature based dynamic time warping. 
International Journal of Computer Science and Network Security, 10(1), 264-273.]
Except the first and the last value of the timeserie. Its local feature is computed
dependend only on the second respectively the penultimate value.
'''
def local_Feature(ts):
    new_timeserie = np.zeros((len(ts),2))
    #set the first and the last feature vector each depending on the follwoing respectively the previous point
    new_timeserie[:][0] = [(ts[0]-ts[1]), (ts[0]-ts[1])]
    new_timeserie[:][-1] = [(ts[-1]-ts[-2]), (ts[-1]-ts[-2])]
    #compute the feature vector of every point in the timeseries according to the definition of Xie et al.
    for i in range(1, len(ts)-1):
        new_timeserie[:][i] = [(ts[i]-ts[i-1]), (ts[i]-ts[i+1])]
    return new_timeserie


'''
Method to compute the global feature of a datapoint in the given time serie, as it is introduced in 
[Xie, Y., & Wiltgen, B. (2010). Adaptive feature based dynamic time warping. 
International Journal of Computer Science and Network Security, 10(1), 264-273.]
Except the first and the last value of the timeserie. Its global feature is computed
dependend only on the second respectively the penultimate value.
'''
def global_Feature(ts):
    new_timeserie = np.zeros((len(ts),2))
    #set the first and the last feature vector each depending on the follwoing 
    #respectively the previous point, additionaly the second, caused by some
    #access problems to vector elements in connection with division with zero in the case of i=1
    new_timeserie[:][0] = [ ts[0] , ts[0]- ((sum(ts[1:])) / float(len(ts)-1)) ] 
    new_timeserie[:][1] = [ ts[1]-ts[0] , ts[1]- ((sum(ts[2:])) / float(len(ts)-1)) ]
    new_timeserie[:][-1] = [ ts[-1]- ((sum(ts[:-1])) / float(len(ts)-1)) , ts[-1] ] 
    #compute the feature vector of every point in the timeseries according to the definition of Xie et al.       
    for i in range(2, len(ts)-1):
        new_timeserie[:][i] = [ ts[i]- (sum(ts[0:i])) / float(i-1) , 
                                ts[i]- float((sum(ts[i+1:])) / float(len(ts)-i)) ]   
    return new_timeserie



#=========================================================================================
# Print accuracy rate and particular confusion matrix of the current classification result
#=========================================================================================

print accuracy_score(vbdtw_set.T[20],vbdtw_set.T[21])  
print confusion_matrix(vbdtw_set.T[20],vbdtw_set.T[21], labels = [1,4,6,5]) 
print classification_report(vbdtw_set.T[20],vbdtw_set.T[21], labels = [1,4,6,5]) 

#print accuracy_score(ddtw_set.T[20],ddtw_set.T[21]) 
#print confusion_matrix(ddtw_set.T[20],ddtw_set.T[21], labels = [1,4,6,5]) 
#print classification_report(ddtw_set.T[20],ddtw_set.T[21], labels = [1,4,6,5]) 
  

#print accuracy_score(fbdtw_set.T[20],fbdtw_set.T[21])  
#print confusion_matrix(fbdtw_set.T[20],fbdtw_set.T[21], labels = [1,4,6,5]) 
#print classification_report(fbdtw_set.T[20],fbdtw_set.T[21], labels = [1,4,6,5]) 


#print accuracy_score(afbdtw_set.T[20],afbdtw_set.T[21])  
#print confusion_matrix(afbdtw_set.T[20],afbdtw_set.T[21], labels = [1,4,6,5])
#print classification_report(afbdtw_set.T[20],afbdtw_set.T[21], labels = [1,4,6,5]) 


#==============================================================================
# Plot the results to visualize the found patterns
#==============================================================================
reload(sys)  
sys.setdefaultencoding('utf8')
#print [plotCategories(6,cnn_data)]#,plotCategories(4,cnn_data),plotCategories(6,cnn_data),plotCategories(5,cnn_data)]


#print [plotCategories(6,skipRepetitions(x_data))]#,plotCategories(4,skipRepetitions(x_data)),plotCategories(6,skipRepetitions(x_data)),plotCategories(5,skipRepetitions(x_data))]
#print len([plotCategories(6,skipRepetitions(x_data))])

# Plot progression of averaged samples of the three different classes to illustrate it in thesis
sampleset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/train_generated.csv", 
                         delimiter = ",", dtype = None, skip_header = 1)
c_means=np.zeros((3,20))
for i in range(0,20):
    c_means[0][i] = np.mean(sampleset.T[i][0:200])
    c_means[1][i] = np.mean(sampleset.T[i][200:400])
    c_means[2][i] = np.mean(sampleset.T[i][500:600])
plt.plot(time_steps,c_means[0], label = "Class 1: Normal Progression")
plt.plot(time_steps,c_means[1], label = "Class 2: Bolus too small")
plt.plot(time_steps,c_means[2], label = "Class 3: Bolus Correction")


plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, ts_length-1, 10, 500])
plt.ylabel('blood glucose content (mg/dL)')
plt.xlabel('timesteps')
plt.show()









#   
#'''
#'''
#def save_overlap_data(data):
#    dyn_timeserie = np.zeros((607,ts_length))
#    #save all possible time series in a new matrix to iterate over all 
#    for i in range(0,607):
#        dyn_timeserie[i][:] = data[i*15:ts_length+(i*15)]
#    new_data = dyn_timeserie 
#    df = pd.DataFrame(new_data)
#    df.to_csv("Data/overlap_25_percent_data.csv",  index=False)  
#
#   

#'''
#Method to claculate the max distances within the particular classes to run the
#weight method.
#'''
#
#def DTWDistance(s1, s2):
#    DTW={}
#    for i in range(-1,len(s1)):
#      for j in range(-1,len(s2)):
#         DTW[(i, j)] = float('inf')
#    DTW[(-1, -1)] = 0
#    for i in range(len(s1)):
#      for j in range(len(s2)):
#         dist= (s1[i]-s2[j])**2
#         DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
#    return np.sqrt(DTW[len(s1)-1, len(s2)-1])
#
#
#'''
#Method to compute the max distance within the classes.
#'''
#def max_Distance(classX, sequence):
#    dists = [-10]
#    for i in enumerate(classX):
#        dists.append(DTW(i,sequence))
#        max_dist = max(dists)
#    return max_dist
#  
#
#        max_class1 =max(dists[0  :99][-1])
#        max_class2 =max(dists[100:199][-1])
#        max_class4 =max(dists[200:299][-1])
#        max_class6 =max(dists[300:399][-1])
#        
#        num_class1 =len(dists[0  :99][-1])
#        num_class2 =len(dists[100:199][-1])
#        num_class4 =len(dists[200:299][-1])
#        num_class6 =len(dists[300:399][-1])
#        
#        
#        if(dists[100:399][-1]<=max_class1):
#            cum1+=1
#        
#        
#        
#        
#        for i in enumerate(dists):
#            for jnd, j in enumerate(dists):
#                jnd +=1
#                if(i[-1]>=j[-1])
#        for time_seq in enumerate(trainset):
#            if(time_seq[-1]==1):
#             class1= np.hstack(class1,time_seq)   
#             max_dist = max_Distance(class1,time_seq)
#             num_class1 =len(class1[0])
#            elif(time_seq[-1]==2):
#             class2= np.hstack(class2,time_seq)
#             max_dist = max_Distance(class2,time_seq)
#            elif(time_seq[-1]==4):
#             class4= np.hstack(class4,time_seq)
#             max_dist = max_Distance(class4)
#            elif(time_seq[-1]==6):
#             class6= np.hstack(class6,time_seq)
#             max_dist = max_Distance(class6, time_seq)
#
#            maxDistclass1 =   
