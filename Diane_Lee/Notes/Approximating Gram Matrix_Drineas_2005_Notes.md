**On the Nystrom Method for Approximating a Gram Matrix for Improved Kernel-Based Learning**
[https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf](https://www.jmlr.org/papers/volume6/drineas05a/drineas05a.pdf)

**Main Approximation Algorithm**
1. Pick c columns of G in i.i.d. Trials, with replacement and with respect to the probabilities (p_i, … p_n)
    a. let I be set of indices of sampled columns
2. Scale each sampled column (index i) by dividing its elements by sqrt(cp_i)
    a. Let C be (nxc) matrix containing rescaled sampled columns
3. Let W be (cxc) submatrix of G with entries G[i,j] / (c*sqrt(p_i*p_j)
4. Compute Wk (best rank-k approximation to W)
5. G_k squiggle = C*W_k^+*C^T

**Nystrom Method**
* Using approximated eigenvalues and eigenvectors to generate low-rank matrix approximations from high-dimensional data
* Speed up computation
* Reduce complexity
