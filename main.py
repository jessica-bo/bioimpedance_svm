# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:13:50 2020

@author: jessi
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

from svm_plotter import Plotter
from data_generator import MakeData


# Generate inverted semi-circle data, then scaling and flipping
#   X = real and imaginary impedance readings
#   y = binary labels
X, y = MakeData.make_data(n_samples=200, noise = 0.3, random_state=None) 
Plotter.plot_points(X, y)

# Split dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,random_state=109)

# Create a SVC classifier using an RBG kernel
svm = SVC(kernel='linear')
#svm = SVC(kernel='rbf', C=10, random_state=0, gamma=0.1)
svm.fit(X_train, y_train)

# Test model predictions 
y_pred = svm.predict(X_test)

# Evaluate overall accuracy of model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", metrics.precision_score(y_test, y_pred))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", metrics.recall_score(y_test, y_pred))

# Visualize the decision boundaries 
#TODO fix this
#Plotter.plot_decision_regions(X_train, y_train, classifier=svm)
#plt.legend(loc='upper left')
#plt.tight_layout()
#plt.savefig('svm.png')
#plt.show()
