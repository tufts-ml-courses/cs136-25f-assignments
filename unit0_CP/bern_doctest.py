'''
Define two ways of computing the Bernoulli log probability mass func.

Tasks
-----
* Complete implementation of forloop and vectorized versions of functions
* Complete doctests, verifying all versions yield same output with same input

Examples, as doctests
---------------------
# Setup printing to just 5 digits after decimal, for simplicity
>>> np.set_printoptions(formatter={'float':lambda x : '%.5f' % x})

# Create some example 0/1 binary data, of size N
>>> prng = np.random.RandomState(101)
>>> N = 50
>>> x_N = prng.choice(2, size=N) # choose 0 or 1, N iid times

# Verify the log likelihood computed when mu = 0.5
>>> calc_bern_log_pmf__A_forloop(x_N, 0.5)
TODO
>>> calc_bern_log_pmf__A_vectorized(x_N, 0.5)
TODO

# Verify mu that favors 0 almost always gives lower likelihood
>>> calc_bern_log_pmf__A_forloop(x_N, 0.01)
TODO
>>> calc_bern_log_pmf__A_vectorized(x_N, 0.01)
TODO

# Verify mu that favors 1 almost always gives lower likelihood
>>> calc_bern_log_pmf__A_forloop(x_N, 0.99)
TODO
>>> calc_bern_log_pmf__A_vectorized(x_N, 0.99)
TODO

'''

import numpy as np


def calc_bern_log_pmf__A_forloop(x_N, mu):
    ''' Compute bernoulli log PMF of dataset at given parameter
    
    Args
    ----
    x_N : 1D array of type int32
        Contains binary values
    mu : scalar float, between 0.0 and 1.0
        Probability of the positive outcome in Bern model
        
    Returns
    -------
    logpmf : np.ndarray, scalar
        Log probability of the entire dataset of N points
    '''
    assert isinstance(mu, float) or isinstance(mu, np.ndarray)
    if mu <= 0.0 or mu >= 1.0:
        raise ValueError("Value of mu must be between 0.0 and 1.0 (exclusive)")
    logpmf = np.array(0.0) # scalar, but array type (for pretty printing)
    N = x_N.size
    for n in range(N):
        if x_N[n] == 1:
            logpmf += np.log(1-mu) # TODO FIXME
        else:
            logpmf += 0.0 # TODO FIXME
    return logpmf
     


def calc_bern_log_pmf__A_vectorized(x_N, mu):
    ''' Compute bernoulli log PMF of dataset at given parameter
    
    Args
    ----
    x_N : 1D array of type int32
        Contains binary values
    mu : scalar float, between 0.0 and 1.0
        Probability of the positive outcome in Bern model
        
    Returns
    -------
    logpmf : np.ndarray, scalar
        Log probability of the entire dataset of N points
    '''
    assert isinstance(mu, float) or isinstance(mu, np.ndarray)
    if mu <= 0.0 or mu >= 1.0:
        raise ValueError("Value of mu must be between 0.0 and 1.0 (exclusive)")

    count1 = np.sum(x_N==1)
    count0 = np.sum(x_N==0)
    logpmf = np.array(0.0) # scalar, but array type (for pretty printing)
    if count1 > 0:
        logpmf += np.log(mu) # TODO FIXME
    if count0 > 0:
        logpmf += count0 # TODO FIXME
    return logpmf
     

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)