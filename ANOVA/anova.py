# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:06:45 2018

@author: hannah
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

datafile="/home/hannah/Dokumente/MTAnalysisOfCGMCurves/ANOVA/Anova_acc.csv"
data = pd.read_csv(datafile)

 
#Create a boxplot
data.boxplot('accuracy', by='classifier', figsize=(12, 8))
 
vb = data['accuracy'][data.classifier == 'VBDTW']
 
grps = pd.unique(data.classifier.values)
d_data = {grp:data['accuracy'][data.classifier == grp] for grp in grps}
 
k = len(pd.unique(data.classifier))  # number of conditions
N = len(data.values)  # conditions times participants
n = data.groupby('classifier').size()[0] #Participants in each condition
	

 
F, p = stats.f_oneway(d_data['VBDTW'], d_data['DDTW'], d_data['FBDTW'], d_data['AFBDTW'], d_data['CNN'])
print F,p
plt.show()


	


