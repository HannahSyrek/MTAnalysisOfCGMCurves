# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:06:59 2018
@author: Hannah syrek
This script implements some needed utilities and parameter to run the classification 
algorithms.
"""

# Needed imports
import numpy as np
import pandas as pd
import sys
import itertools
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
trainset = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/trainset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1)
labeled_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Generator/testset.csv", 
                         delimiter = ",", dtype = None, skip_header = 1)                         
_data = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/export-v2.csv",
                         delimiter = ",", dtype = None, skip_header = 1, filling_values = -1, usecols = [3])
ddtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/DDTW_labeled.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
vbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/VBDTW_labeled_test.csv", 
                        delimiter = ",", dtype = None, skip_header = 1) 
afbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/AFBDTW_labeled_trainset.csv", 
                        delimiter = ",", dtype = None, skip_header = 1)
fbdtw_set = np.genfromtxt("/home/hannah/Dokumente/MTAnalysisOfCGMCurves/Classifier/Data/fbdtw_labeled_trainset.csv", 
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


"""
This function prints and plots the confusion matrix of the applied classifier.
"""
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, normalize=False):
    if normalize:
        cm_temp = cm
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")  
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm_temp[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#==============================================================================
# Print accuracy rate and particular classificaton report
#==============================================================================

#print accuracy_score(vbdtw_set.T[20],vbdtw_set.T[21])  
#print classification_report(vbdtw_set.T[20],vbdtw_set.T[21], labels = [1,4,6,5]) 

#print accuracy_score(ddtw_set.T[20],ddtw_set.T[21]) 
#print classification_report(ddtw_set.T[20],ddtw_set.T[21], labels = [1,4,6,5]) 

#print accuracy_score(fbdtw_set.T[20],fbdtw_set.T[21])  
#print classification_report(fbdtw_set.T[20],fbdtw_set.T[21], labels = [1,4,6,5]) 

#print accuracy_score(afbdtw_set.T[20],afbdtw_set.T[21])  
#print classification_report(afbdtw_set.T[20],afbdtw_set.T[21], labels = [1,4,6,5]) 

#==============================================================================
# Plot the results 
#==============================================================================
reload(sys)  
sys.setdefaultencoding('utf8')
# Plot found patterns 
#print plotCategories(6,cnn_data)
plt.plot(time_steps, upper_bound,'r--', time_steps, lower_bound, 'r--')
plt.legend(loc=1)
plt.axis([0, ts_length-1, 10, 500])
plt.ylabel('blood glucose content (mg/dL)')
plt.xlabel('timesteps')

# Compute confusion matrix
vbdtw_cnf_matrix= confusion_matrix(vbdtw_set.T[20],vbdtw_set.T[21],labels = [1,4,6,5]) 
ddtw_cnf_matrix= confusion_matrix(ddtw_set.T[20],ddtw_set.T[21],labels = [1,4,6,5])
fbdtw_cnf_matrix= confusion_matrix(fbdtw_set.T[20],fbdtw_set.T[21],labels = [1,4,6,5]) 
afbdtw_cnf_matrix= confusion_matrix(afbdtw_set.T[20],afbdtw_set.T[21],labels = [1,4,6,5]) 
np.set_printoptions(precision=2)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(fbdtw_cnf_matrix, classes=['1','2','3','4'], title='Confusion matrix of the FBDTW',normalize=True)
plt.show()




