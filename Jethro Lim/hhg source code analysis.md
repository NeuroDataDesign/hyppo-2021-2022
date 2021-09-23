**hhg.py source code analysis**

**HHG Init function:**
- If compute_distance is empty, is_distance is True. Else, is_distance is False.
  - Is_distance = True is for indicating that the input matrices are distance matrices, rather than data matrices
- Then initiates Independence Test parent class.

**HHG statistic function - Helper Function:**
- Needs (x, y) which are the data matrices/samples 
- If is_distance is False, runs compute_dist from hyppo.tools
  - Gives back the distance matrices among the samples within each data matrix
- Then places distance matrices in _pearson_stat 
  - Gives back Pearson chi square statistics
  - Here it performs the centre-point-random-point comparison mentioned in HHG
- Creates a NxN boolean mask with all 1s
- Then converts the diagonal in the mask to 0s.
  - This is because no pearson statistic computation is done for the diagonal scenario (where the centre point and the distance point are the same).
- Then takes the sum of all the Pearson correlation as the test statistic.

**HHG test function - primary called function:**
- Accepts data matrices x/y, number of repetitions and number of workers
- Puts input through _CheckInputs from hyppo.independence.utils
  - Used to convert x and y to proper dimensions
- Then runs x and y through compute_distance to get distance matrices using the computation method specified in __init__
- Changes is_distance to True to avoid statistic function running computation again
- Uses super(HHG, self) - essentially same as super() - to run the permutation test, which also contains the test-statistic function described above.
- Gives back independence test statistic and p-value.

**_pearson_stat function:**
- Takes xdist, ydist distance matrices
- Creates an N-by-N matrix where N is the number of samples in either x or y
- Iterates through each point in X and Y, choosing two points as a centre point and a random point to create a radial distance, and comparing the distance between the centre and other points to the radial distance
- Creates a Pearson chi square that then has its statistic computed. Statistic is returned to caller.
