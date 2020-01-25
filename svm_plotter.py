# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:45:06 2020

@author: jessi
"""
# Import packages to visualize the classifer
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import numpy as np

class Plotter():

    def versiontuple(v):
        return tuple(map(int, (v.split("."))))
    
    def plot_points(X,y):
        plt.scatter(X[y == 1, 0],
            X[y == 1, 1],
            c='b', marker='x',
            label='1')
        
        plt.scatter(X[y == 0, 0],
            X[y == 0, 1],
            c='r',marker='s',
            label='0')
        
#        plt.xlim([-3, 3])
#        plt.ylim([-3, 3])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
    
    
    #TODO fix this 
    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
        
        # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        xy = np.array([xx1.ravel(), xx2.ravel()]).T
        xyf = np.hstack([xy, np.vstack(100*np.ones(len(xy)))])
        Z = classifier.decision_function(xyf).reshape(xx1.shape)
        
#        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#                               np.arange(x2_min, x2_max, resolution))
#        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
    
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=cmap(idx),
                        marker=markers[idx], label=cl)
    
        # highlight test samples
        if test_idx:
            # plot all samples
            if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
                X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
                warnings.warn('Please update to NumPy 1.9.0 or newer')
            else:
                X_test, y_test = X[test_idx, :], y[test_idx]
    
            plt.scatter(X_test[:, 0],
                        X_test[:, 1],
                        c='',
                        alpha=1.0,
                        linewidths=1,
                        marker='o',
                        s=55, label='test set')
