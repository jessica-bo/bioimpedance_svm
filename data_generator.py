# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 15:31:13 2020

@author: jessi
"""

import numpy as np

class MakeData(): 
    
    def make_data(n_samples=100, noise=0.1, random_state=None):

        n_samples_outer = n_samples // 2
        n_samples_inner = n_samples - n_samples_outer
    
        outer_circ_x = np.linspace(0, np.pi, n_samples_outer) + np.random.normal(0, noise, n_samples_outer)
        outer_circ_y = np.sin(outer_circ_x) + np.random.normal(0, noise, n_samples_outer)
        outer_freqs = np.linspace(100, 10000, n_samples_outer)
        
        inner_circ_x = np.linspace(0, 2*np.pi, n_samples_inner) + np.random.normal(0, noise, n_samples_inner)
        inner_circ_y = 2*np.sin(1/2* inner_circ_x) + np.random.normal(0, noise, n_samples_inner)
        inner_freqs = np.linspace(100, 10000, n_samples_inner) 
    
        X = np.hstack([np.vstack([np.append(outer_circ_x, inner_circ_x),
                       np.append(outer_circ_y, inner_circ_y)]).T, 
                        np.vstack(np.append(outer_freqs, inner_freqs))])
    
        y = np.hstack([np.zeros(n_samples_outer, dtype=np.intp),
                       np.ones(n_samples_inner, dtype=np.intp)])
    
        return X, y