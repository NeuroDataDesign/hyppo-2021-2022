import numpy as np
from math import ceil

from hyppo.tools import indep_sim
import pandas as pd

df = pd.read_csv('housing.csv')
data = df[['RM', 'MEDV']].copy()

def _indep_sim_gen(sim, n, p, noise=True):
    """
    Generate x, y from conditional dataset
    """
    subset = data.sample(n=n)
    x = np.array(subset['RM'].tolist())[np.newaxis].T
    y = np.array(subset['MEDV'].tolist())[np.newaxis].T

    return x, y


def _perm_stat(test, sim, n=100, p=1, noise=True):
    """
    Generates null and alternate distributions
    """
    x, y = _indep_sim_gen(sim, n, p, noise=noise)
    obs_stat = test().statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test().statistic(x, permy)

    return obs_stat, perm_stat


def power(test, sim, n=100, p=1, noise=True, alpha=0.05, reps=1000, auto=False):
    """
    Calculates empirical power
    """
    alt_dist, null_dist = map(
        np.float64,
        zip(*[_perm_stat(test, sim, n, p, noise=noise) for _ in range(reps)]),
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
