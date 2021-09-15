**A Kernel Statistical Test for Independence (Gretton et al. 2007)**

[https://papers.nips.cc/paper/2007/file/d5cfead94f5350c12c322b5b664544c1-Paper.pdf](https://papers.nips.cc/paper/2007/file/d5cfead94f5350c12c322b5b664544c1-Paper.pdf)

**Hilbert-Schmidt Independence Criterion (HSIC)**

* Sum of squared singular values
* Fancy F and fancy G are universal reproducing kernel Hilbert spaces
* K and l are kernels in F and G

**HSIC Statistic**

* Def: HSIC_b(Z) = 1/(m^2) * trace(KHLH)
    * Z: sample
    * M: tuples drawn with replacement from {1...m}
    * K, L: kernel matrices
    * H: I - 1/m*1(vec)*1(vec)^T
* E(HSIC_b(Z)) = 1/m * trace(C_xx) * trace(C_yy)
    * C_xy: cross covariance matrix between random vectors
    * var(HSIC_b(Z)) = 2(m-4)(m-5)/m^4 * ||C_xx||^2 * ||C_yy||^2 + O(m^-3)

**Test Description**

* Derive asymptotic distribution of HSIC_b(Z) under null
    * HSIC_b(Z) converges according to sum(lambda_l * z_l^2) from l = 0 to inf
      * z_l ~ N(0,1) i.i.d. 
      * lambdas are eigenvalues 
    * Use (1-alpha) quantile distribution as test threshold
      * Approximated null distribution as two-parameter Gamma distribution
