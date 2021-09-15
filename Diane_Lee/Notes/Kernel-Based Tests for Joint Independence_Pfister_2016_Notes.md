**Kernel-based Tests for Joint Independence (Pfister et al. 2016) [https://arxiv.org/abs/1603.00285](https://arxiv.org/abs/1603.00285)**

**Algorithm / Procedure**

* Objective: determine if two distributions are jointly independent
1. For a given positive definite kernel (normally Gaussian), embed distributions into appropriate reproducing kernel Hilbert space (RKHS)
2. Calculate/estimate d-variable Hilbert-Schmidt independence criterion (dHSIC)
    1. Similar to L^2 distance between ‘traditional’ kernel density estimators
    2. Algorithm 1 (pg. 13)
    3. Hypothesis tests: Algorithm 2 (pg. 33)
        - Permutation test (Monte-Carlo approximated)
        - Bootstrap test (Monte-Carlo approximated)
        - Gamma approximation based test
        - Eigenvalue based test*
            1. Implemented in R package but not included in paper because it performed worse than Gamma approximation in almost all experiments?

**Experiments**

* Competing methods
    * BMR-C
    * Pairwise HSIC d-1 times + Bonferroni correction
* Testing: plotted rejection rates vs. sample size n
    * Level analysis
        * three continuous (m=1000)
        * continuous and discrete (m=1000)
    * Power analysis
        * single edge (m=1000)
        * full DAG (n=100, m=1000)
        * dense and sparse (m=10000)
            * Rejection rates vs. total variation distance
        * Bandwidth of Gaussian kernel (m=1000)
            * Rejection rates vs. bandwidth
    * Runtime analysis
        * dHSIC: O(dn^2)
        * HSIC d-1 times: O(d^2*n^2)
    * Casual inference
        * DAG verification method
            * Use generalized additive model regression (GAM) ([Wood and Augustin, 2002](https://www.sciencedirect.com/science/article/abs/pii/S030438000200193X?via%3Dihub)) to regress each node on all its parents and denote resulting residual vector (res)
            * Perform dHSIC to test (res1,...resd) is jointly independent
            * f (res1,...resd) is jointly independent, candidate DAG is not rejected
            
        * Simulated data: Gaussian distributed noise variables, DAG candidate over d=4 nodes
            * Compared gamma dHSIC and pairwise HSIC
        * Real data: 25 possible DAGs
            * Compared permutation dHSIC and BMR-1000
