**OG HHG**

**Paper:**  A consistent multivariate test of association based on ranks of distances (2013)
**Implementation:** https://hyppo.neurodata.io/api/generated/hyppo.independence.hhg#hellerconsistentmultivariatetest2013 

**Principle**

1) Computes the pairwise distance between sample values of X and Y respectively (i.e. calculating distance between all possible pairs within the X sample and Y sample)
    1) In simple terms, in each ‘round’, we pick two sample points - one center (xi, yi) and one radii point (xj, yj). The distance between these points are computed to give us a radii. We then see how the distance between the centre points and other sample points compare to this radii distance in both the x and y vector dimension, giving us a 2x2 cross-classification table.
    2) Distance metrics are determined by norms
2) Using the 2x2 cross-classification table, we can then compute the summation of the Pearson’s chi square test statistics for all i and for j, giving us T.
3) T is then put through permutation tests in order to find the p-value.
    1) Based on the fraction of replicates of T under random permutations of the indices of the Y sample, that are at least as large as the observed statistic.


**Fast HHG**

**Paper:** Multivariate tests of association based on univariate tests (2016)

**Principle**

1) Pick a center point - choices can be a sample point, all the sample points or center of mass. Then calculate the distance between this center point and every single sample point (except itself of course)
Unlike OG, there is no 2x2 cross-classification based on distance comparison. Instead you accumulate the distances into a sample.
2) Put the sample of distances through a univariate consistent test
    1) Take note of the assumptions of the distribution of the distance of the multivariate vectors 
    2) Best to use tests with a known asymptotic null distribution
    3) Can give nice to use test statistic that converges to easily interpretable population values between 0 and 1
3) At this point, you can:
    1) compute the significance from the asymptotic null distribution
    2) OR if you did multiple center points, combine the univariate test statistics of each point into a consistent multivariate test.
3 approaches detailed in fast HHG paper notes
