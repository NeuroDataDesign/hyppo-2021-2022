**Overall Notes for Design:**

**Basic Approach** - compute distance from center point and apply univariate independence tests on distance
**Goal** - achieve multivariate consistent independence test through univariate consistent test
- Consistent = power increasing to 1 as sample size increases when null is false
- Aim to achieve fast computational speed due to univariate testing

**Pooling** - where M center points are defined, resulting in M univariate tests. Test results can then be pooled into a test 
- Suggested Method: apply a multiple comparison test that tests the global null hypothesis that all M null hypotheses are true

**Pooling Alternatives:** 1) Pooling of p-values 2) Pooling of test-statistics

Both alternativess then tested by permutation tests

Consequence of alts: lose distribution-free property of univariate test, but expect more power

**Choice of Center Point(s)** - possibly set in code with three settings shown in paper:
1) Singular Center mass
2) Singular Random Sample Points
3) All Sample Points = Center Points -> pooling

**Tests to use** - desirable tests are distribution-free and have known asymptotic null distribution
**Univariate tests **
1) KS
2) Anderson-Darling (AD)
3) minP = generalized test that aggregates over all partition sizes using min p-value setting (described more in supplemental material)
**Multiple Comparison Tests**
1) Bonferroni
2) Hommel

**Paper Author Recommendations:** 
At least one centre point randomly sampled from a distribution with a support of positive measure
Want to avoid picking a centre point that converges to a bad point
Adding Gaussian error to the measured signals guarantees that the results from fast HHG hold for any normed distance.
Without Gaussian, results shown to hold in Euclidean norm.

**Notable Experimental Results:**
Benefit in considering all sample points as center vs single center point
Between Hommel and Bonferroni, latter was found to have better power in most scenarios
Between univariate tests, AD and minP are found to be more powerful than KS, with minP gaining more in many-clustered data
If the univariate test-statistic is a U-statistic of order m, then aggregating by summation with the sample points as center points produces a multivariate test-statistic which is a U-statistic of order m + 1.
Useful in working out asymptotic null distribution of multivariate test-statistic or identify non-null distribution of test-statistic

**SUPPLEMENTAL MATERIALS**

Contains proofs of corollaries and theorems presented in paper. Math is quite beyond me - need to consult with Sampan on relevance.

**Additional Experiments**

Between distributions of high dimension, power of test deteriorated with increase in vector dimension in all approaches, with the minP approach as a univariate demonstrating greater power.

When sample size is increased, taking all sample point as center points resulted in even greater power.

**Further Description of Univariate Tests used in Paper**

Assume Y is a univariate continuous random variable and X is categorical with K>2 categories, with N independent realizations from joint distributions of X and Y.

KS, AD and CM tests look at all possible ‘partitions of yi’s into two parts’ and give a score to each partition - they aggregate these scores by maximization or summation.

In (2): ‘Consistent distribution-free K-sample and independence tests for univariate random variables’, they instead aggregate scores for all possible partitions of the data, not just into two parts.

minP combines the p-values from partitioning into m number of cells through either summation or maximization (2). It takes the minimum of the p-values obtained through partitioning into various number of cells (up to m’max):

