**A Fast, Consistent Kernel Two-Sample Test (Gretton et al. 2009)**

[https://proceedings.neurips.cc/paper/2009/file/9246444d94f081e3549803b928260f56-Paper.pdf](https://proceedings.neurips.cc/paper/2009/file/9246444d94f081e3549803b928260f56-Paper.pdf)

**Approximation of Null Distribution**

* Prior methods
    * Bootstrap resampling: consistent but computationally costly
    * Fitting parametric model with low order moments of test stat: no consistency or accuracy guarantees (works well in practice though)
* Novel estimate of null distribution 
    * Computed from eigenspectrum of Gram matrix on aggregate sample from two probability measures P and Q

**Test Method**

* Covariance operator C: F->F
  * var(f(x)) 
* Eigenvalues of C estimated by:
  * 1/m * v_l where v_l are eigenvalues of the centered Gram matrix
* Centered Gram matrix: K squiggle := HKH
  * K is kernel 
  * H is centering matrix I - 1/m * 1 * 1T
* Maximum mean discrepancy (MMD) (aka distance between mappings of P and Q to RKHS F) estimated by:
  * sum(lambda_l * z_l^2) for l = 0 to inf
    * Z_l = (x_l, y_l)
    * -> D means convergence in distribution
