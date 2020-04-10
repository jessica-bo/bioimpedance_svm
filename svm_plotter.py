# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:45:06 2020

@author: jessicabo
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
import numpy as np

class Plotter():

    def versiontuple(v):
        return tuple(map(int, (v.split("."))))
    
    def plot_points(X, y, var1, var2):
        fig, ax = plt.subplots()
        colors = {1:'red', 0:'blue'}
        ax.scatter(X[var1], X[var2], c=y['Blood?'].apply(lambda x: colors[x]))
#        ax.set_yscale('log')
        plt.tight_layout()
        plt.title("Imag vs. Real")
        plt.savefig('data.png')
        plt.show()

    def plot_decision_regions(X, y, classifier, var1, var2, resolution=100, test_idx=None):
        # setup marker generator and color map
        colors = ('blue', 'red','lightgreen', 'gray', 'cyan')
        y_colors = {1:'red', 0:'blue'}
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # plot the decision surface
        x1_min, x1_max = X[var1].min() - 1, X[var1].max() + 1
        x2_min, x2_max = X[var2].min() - 1, X[var2].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max-x1_min)/5000), 
                               np.arange(x2_min, x2_max, (x2_max-x2_min)/5000))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        fig = plt.figure()
        ax = plt.gca()
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.scatter(X[var1], X[var2], c=y['Blood?'].apply(lambda x: y_colors[x]))
    
        # highlight test samples
        if test_idx:
            # plot all samples
            if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
                X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
                warnings.warn('Please update to NumPy 1.9.0 or newer')
            else:
                X_test, y_test = X[test_idx, :], y[test_idx]
    
            plt.scatter(X_test[:, 0],X_test[:, 1],c='',alpha=1.0,linewidths=1,
                        marker='o',s=55, label='test set')
            
        plt.title("SVM Decision Boundary")
#        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('svm.png')
        plt.show()
