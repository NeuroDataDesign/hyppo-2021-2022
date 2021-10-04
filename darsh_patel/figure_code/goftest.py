"""
Module that will contain several goodness-of-fit test methods
"""

from __future__ import division

from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass

__author__ = 'patel'

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
# import kgof.data as data
# import kgof.util as util
# import kgof.kernel as kernel
import logging
import matplotlib.pyplot as plt

import scipy
import scipy.stats as stats

class GofTest(with_metaclass(ABCMeta, object)):
    """
    Abstract class for a goodness-of-fit test.
    """

    def __init__(self, p, alpha):
        """
        p: an UnnormalizedDensity
        alpha: significance level of the test
        """

        self.p = p
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, dat):
        """
        perform the GoF test and return values computed in a dictionary:
        {
            alpha: 0.01,
            pvalue: 0.0002,
            test_stat: 2.3,
            h0_rejected: True,
            time_secs: ...
        }

        dat: instance of Data
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, dat):
        """Compute the test statistic"""
        raise NotImplementedError()
    
# end of GofTest
#-----------------------------------------------------------------------

class H0Simulator(with_metaclass(ABCMeta, object)):
    # Simulator to draw samples from the null distribution
    x = 3

# end of H0Simulator
#-----------------------------------------------------------------------

class FSSDH0SimCovObs(H0Simulator):
    # Asymptotic null distribution simulator for FSSD
    x = 3

# end of FSSDH0SimCovObs
#-----------------------------------------------------------------------

class FSSDH0SimCovDraw(H0Simulator):
    """
    Asymptotic null distribution simulator for FSSD. The eigenvalues are computed from
    the covariance matrix wrt. the sample drawn from p (the density to test against)
    """
    x = 3

# end of FSSDH0SimCovDraw
#-----------------------------------------------------------------------

class FSSD(GofTest):
    """
        GoF test using the Finite Set Stein Discrepancy statistic.
        and a set of paired test locations. The statistic is n*FSSD^2.
        The statistic can be negative because of the unbiased estimator.

        H0: the sample follows p
        H1: the sample does not follow p

        p is specified to the constructor in the form of an UnnormalizedDensity
    """

    #NULLSIM_* are constants used to choose the way to simulate from the null 
    #distribution to do the test.


    # Same as NULLSIM_COVQ; but assume that sample can be drawn from p. 
    # Use the drawn sample to compute the covariance.
    NULLSIM_COVP = 1


    def __init__(self, p, k, V, null_sim=FSSDH0SimCovObs(n_simulate=3000,
        seed=101), alpha=0.01):
        """
        p: an instance of UnnormalizedDensity
        k: a DifferentiableKernel object
        V: J x dx numpy array of J locations to test the difference
        null_sim: an instance of H0Simulator for simulating from the null distribution.
        alpha: significance level 
        """
        super(FSSD, self).__init__(p, alpha)
        self.k = k
        self.V = V 
        self.null_sim = null_sim
    
    def perform_test(self, dat, return_simulated_stats=False):
        """
        dat: an instance of Data
        """
        with util.ContextTimer() as t:
            alpha = self.alpha
            null_sim = self.null_sim
            n_simulate = null_sim.n_simulate
            X = dat.data()
            n = X.shape[0]
            J = self.V.shape[0]

            nfssd, fea_tensor = self.compute_stat(dat, return_feature_tensor=True)
            sim_results = null_sim.simulate(self, dat, fea_tensor)
            arr_nfssd = sim_results['sim_stats']

            # approximate p-value with the permutations 
            pvalue = np.mean(arr_nfssd > nfssd)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': nfssd,
                'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                'time_secs': t.secs, 
                }
        if return_simulated_stats:
            results['sim_stats'] = arr_nfssd
        return results
    
    def compute_stat(self, dat, return_feature_tensor=False):
        """
        The statistic is n*FSSD^2.
        """
        X = dat.data()
        n = X.shape[0]

        # n x d x J
        Xi = self.feature_tensor(X)
        unscaled_mean = FSSD.ustat_h1_mean_variance(Xi, return_variance=False)
        stat = n*unscaled_mean


        if return_feature_tensor:
            return stat, Xi
        else:
            return stat
    
    def get_H1_mean_variance(self, dat):
        """
        Return the mean and variance under H1 of the test statistic (divided by
        n).
        """
        X = dat.data()
        Xi = self.feature_tensor(X)
        mean, variance = FSSD.ustat_h1_mean_variance(Xi, return_variance=True)
        return mean, variance

    def grad_log(self, X):
        """
        Evaluate the gradients (with respect to the input) of the log density at
        each of the n points in X. This is the score function. Given an
        implementation of log_den(), this method will automatically work.
        Subclasses may override this if a more efficient implementation is
        available.
        X: n x d numpy array.
        Return an n x d numpy array of gradients.
        """
        g = autograd.elementwise_grad(self.log_den)
        G = g(X)
        return G

    def outer_rows(X, Y):
        """
        Compute the outer product of each row in X, and Y.
        X: n x dx numpy array
        Y: n x dy numpy array
        Return an n x dx x dy numpy array.
        """

        #n, dx = X.shape
        #dy = Y.shape[1]
        #X_col_rep = X[:, np.tile(range(dx), (dy, 1)).T.reshape(-1) ]
        #Y_tile = np.tile(Y, (1, dx))
        #Z = X_col_rep*Y_tile
        #return np.reshape(Z, (n, dx, dy))
        return np.einsum('ij,ik->ijk', X, Y)

    @abstractmethod
    def dim(self):
        """
        Return the dimension of the input.
        """
        raise NotImplementedError()

    def feature_tensor(self, X):
        """
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.
        X: n x d data numpy array
        return an n x d x J numpy array
        """
        k = self.k
        J = self.V.shape[0]
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        #assert np.all(util.is_real_num(grad_logp))
        # n x J matrix
        #print 'V'
        #print self.V
        K = k.eval(X, self.V)
        #assert np.all(util.is_real_num(K))

        list_grads = np.array([np.reshape(k.gradX_y(X, v), (1, n, d)) for v in self.V])
        stack0 = np.concatenate(list_grads, axis=0)
        #a numpy array G of size n x d x J such that G[:, :, J]
        #    is the derivative of k(X, V_j) with respect to X.
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = FSSD.outer_rows(grad_logp, K)
        #print 'grad_logp'
        #print grad_logp.dtype
        #print grad_logp
        #print 'K'
        #print K
        Xi = old_div((grad_logp_K + dKdV),np.sqrt(d*J))
        #Xi = (grad_logp_K + dKdV)
        return Xi

    @staticmethod
    def power_criterion(p, dat, k, test_locs, reg=1e-2, use_unbiased=True, use_2terms=False):
        """
        Compute the mean and standard deviation of the statistic under H1.
        Return mean/sd.
        use_2terms: True if the objective should include the first term in the power 
            expression. This term carries the test threshold and is difficult to 
            compute (depends on the optimized test locations). If True, then 
            the objective will be -1/(n**0.5*sigma_H1) + n**0.5 FSSD^2/sigma_H1, 
            which ignores the test threshold in the first term.
        """
        X = dat.data()
        n = X.shape[0]
        V = test_locs
        fssd = FSSD(p, k, V, null_sim=None)
        fea_tensor = fssd.feature_tensor(X)
        u_mean, u_variance = FSSD.ustat_h1_mean_variance(fea_tensor,
                return_variance=True, use_unbiased=use_unbiased)

        # mean/sd criterion 
        sigma_h1 = np.sqrt(u_variance + reg)
        ratio = old_div(u_mean,sigma_h1) 
        if use_2terms:
            obj = old_div(-1.0,(np.sqrt(n)*sigma_h1)) + np.sqrt(n)*ratio
            #print obj
        else:
            obj = ratio
        return obj

    def ustat_h1_mean_variance(fea_tensor, return_variance=True, use_unbiased=True):
        """
        Compute the mean and variance of the asymptotic normal distribution 
        under H1 of the test statistic.
        fea_tensor: feature tensor obtained from feature_tensor()
        return_variance: If false, avoid computing and returning the variance.
        use_unbiased: If True, use the unbiased version of the mean. Can be
            negative.
        Return the mean [and the variance]
        """
        Xi = fea_tensor
        n, d, J = Xi.shape
        #print 'Xi'
        #print Xi
        #assert np.all(util.is_real_num(Xi))
        assert n > 1, 'Need n > 1 to compute the mean of the statistic.'
        # n x d*J
        # Tau = Xi.reshape(n, d*J)
        Tau = np.reshape(Xi, [n, d*J])
        if use_unbiased:
            t1 = np.sum(np.mean(Tau, 0)**2)*(old_div(n,float(n-1)))
            t2 = old_div(np.sum(np.mean(Tau**2, 0)),float(n-1))
            # stat is the mean
            stat = t1 - t2
        else:
            stat = np.sum(np.mean(Tau, 0)**2)

        if not return_variance:
            return stat

        # compute the variance 
        # mu: d*J vector
        mu = np.mean(Tau, 0)
        variance = 4*np.mean(np.dot(Tau, mu)**2) - 4*np.sum(mu**2)**2
        return stat, variance

    # end of FSSD
    #----------------------------------------------------------------------
        

