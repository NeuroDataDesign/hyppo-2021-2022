# A fast algorithm for computing distance correlation
- Distance covariance and its derived correlation
  - Covariance is a weighted L2 distance between joint characteristic function and the product of marginal distributions
  - To calculate sample distance covariance between two univariate random variables O(nlog(n)) algorithm is developed
    - Different from most O(n log(n)) algorithms where this paper's method is valid for any pair of real-valued univariate variables
    - They also use a merge sort instead of an AVL tree-type implementation to compute the Frobenius inner product of the distance matrixes of x and y

## Definition of distance covariance and it's sample estimate
![image](https://user-images.githubusercontent.com/89371970/134945878-2c1d7212-4412-43b2-9988-51886a4b30e3.png)
Provides detailed knowledge of this, but specifities might not be needed

## Details on Fast algorithms for multivariate random varaibles and how to Calculate D/What is it
![image](https://user-images.githubusercontent.com/89371970/134946092-d67f83ca-0103-4cee-b0a0-86f7a775e19c.png)
![image](https://user-images.githubusercontent.com/89371970/134946109-0c769f75-bcc3-4208-870f-a04bcf972b37.png)

## In the end O(n log(n)) algorithm is displayed for sample distance covariance
![image](https://user-images.githubusercontent.com/89371970/134946197-57c410f8-3f9d-4387-b32e-7585461084b9.png)
