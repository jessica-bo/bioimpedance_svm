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
        fig, ax = plt.subplots()
        colors = {1:'red', 0:'blue'}
        ax.scatter(X['Frequency'], X['Impedance'], c=y['Blood?'].apply(lambda x: colors[x]))
        plt.tight_layout()
        plt.title("Impedance vs. Frequency")
        plt.savefig('data.png')
        plt.show()
    
    #TODO fix this 
    def plot_decision_regions(X, y, classifier, test_idx=None, resolution=100):
        
        # setup marker generator and color map
#        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('blue', 'red','lightgreen', 'gray', 'cyan')
        y_colors = {1:'red', 0:'blue'}
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # plot the decision surface
        x1_min, x1_max = X['Frequency'].min() - 1, X['Frequency'].max() + 1
        x2_min, x2_max = X['Impedance'].min() - 1, X['Impedance'].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, (x1_max-x1_min)/5000), 
                               np.arange(x2_min, x2_max, (x2_max-x2_min)/5000))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        plt.scatter(X['Frequency'], X['Impedance'], c=y['Blood?'].apply(lambda x: y_colors[x]))
    
#        colors = {1:'red', 0:'blue'}
#        
#        for idx, cl in enumerate(np.unique(y)):
#            plt.scatter(x=X['Frequency'], y=X['Impedance'],
#                        alpha=0.8, c=cmap(idx),
#                        marker=markers[idx], label=cl)
    
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
            
        plt.title("SVM Decision Boundary")
        plt.tight_layout()
        plt.savefig('svm.png')
        plt.show()
