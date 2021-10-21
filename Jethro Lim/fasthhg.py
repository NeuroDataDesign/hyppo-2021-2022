#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import jit

from ..tools import check_perm_blocks_dim, chi2_approx, compute_dist
from ._utils import _CheckInputs
from .base import IndependenceTest
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp, cramervonmises_2samp



class FastHHG(IndependenceTest):
    
    def __init__(self, compute_distance="euclidean", bias=False, **kwargs):
        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        IndependenceTest.__init__(self, compute_distance=compute_distance, **kwargs)
    
    def test(self, x, y, point, unitest = 'KS'):
        r"""
        Calculates the Fast HHG test statistic
        
        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` do not need to have the 
            same number of samples.
            
        point : ndarray or string
            Single center point used for distance calculations from sample
            points. If ndarray, must be in the form of [zx, zy], where zx
            is a point in the space of x and zy is a point in the space of y.
            If string, must 
        
        unitest : string
        
        """
        check_input = _CheckInputs(
            x,
            y
        )
        x, y = check_input()
        
        distx = x
        disty = y
        
        zx, zy = point
        
        if not (self.is_distance):
            distx, disty = self._point_distance(x, y, zx, zy)
        
        if unitest == 'KS':
            stat, pvalue = ks_2samp(distx, disty)
            
        elif unitest == 'CM':
            stat, pvalue = cramervonmises_2samp(distx, disty)
        
        return stat, pvalue
        
        
    def _point_distance(x, y, zx, zy, workers=1, **kwargs):
        """
        Returns a collection of distances between sample points and chosen centre point
        """
        distx = pairwise_distances(x, zx, metric=self.compute_distance, n_jobs=workers, **kwargs)
        disty = pairwise_distances(y, zy, metric=self.compute_distance, n_jobs=workers, **kwargs)
        
        return distx, disty
        
        

