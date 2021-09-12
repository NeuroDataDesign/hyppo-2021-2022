## Discovering and deciphering relationships across disparate data modalities
Discovering and deciphering relationships across disparate data modalities (a description of the process of MGC is the main desire here)

MGC: a dependance test that juxtaposes disparate data science techniques including k-nearest neighbors, kernel methods, and multiscale analysis
MGC procedure: given n samples of two different properties
- Compute distance matrices
  - One consisting distance between all pairs of one property (cloud densities)
  - Other consisting of distance between all pairs of other property
  - Center each distance matrix (by subtracting overall mean, column-wise mean from each column, and the row-wise mean from each row)
    - Results in a n xn matrix labelled as A and B
- Then find all values for k and l (a scale from 1 to n)
  - ![image](https://user-images.githubusercontent.com/89371970/132999984-9dc89d69-5f59-4eb8-97fc-8b5576029189.png)
- Next step is to estimate optimal local correlation c* by finding smoothed maximum of matrix above
  - Smoothing mitigates biases and provides MGC with theoretical guarantee and better finite-sample performance
- Determine if relationship is important: "whether c* is more extreme than under the null hypothesis" by doing permutation test
  - Permutation procedure basically repeats the steps stated above repeat, this helps in computation process eliminating need to calculate overall p-value instead of one p-value per scale

Benchmark Test img below to show power compared to other statistical tests
![image](https://user-images.githubusercontent.com/89371970/133000152-61b902e0-a07f-4e8f-bbd6-43c2368a1433.png)

The MGC map: shows local correlation as a function of scales of two properties
- In essence, it's the matrix of ckl's. MGC map is basically a n-byn- matrix which encodes strength of dependence for each possible scale
![image](https://user-images.githubusercontent.com/89371970/133000572-5ab5f404-80b8-4dcc-a3a6-dc743b551334.png)

