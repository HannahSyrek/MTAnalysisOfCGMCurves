# MTAnalysisOfCGMCurves
# Author: Hannah Syrek
Statistical analysis of cgm curves to find certain patterns.


Tiel:
Classifying CGM Curves Using a
Deep Neural Network to Improve
Therapy Decisions of Diabetes
Type I Patients


Abstract:
In this thesis, special emphasis will be placed on the investigation of five different ways of
classifying continuous glucose monitoring (CGM) curves of diabetes type I patients using an
insulin pump to regulate their metabolism. The implemented Convolutional Neural Network
(CNN) was examined for its convenience and performance in classifying time series compared to
the commonly used dynamic time warping (DTW) algorithm.
In addition to classic DTW, three modified alternatives containing manually extracted features in
the distance calculation were implemented. Prior to the examination of the different models,
four individual glucose trends are specified, representing the classes to be recognized.
In comparison with the four different methods using DTW, the CNN yields significantly better
results in recognizing the four CGM patterns with an accuracy rate of 0.72 on a labelled test set. A
reason for the good results of the CNN could be its special operating principle in the classification,
based on a mechanical feature extraction. The input data is not recognized as independent
elements, but arises from a spatial structure. The application of CNNs to observe structural
patterns in physiological time series is a promising solution that requires further research.


This project implements a CNN and four different classifiers that accroding to the commonly used 
dynamic time warping algorithm. The main goal is to classify real-world data of diabetes type I 
patients recognizing certain patterns in the given data sets. Needed mathematical and further 
background information of the implemted classifiers are released in the submitted master thesis.




