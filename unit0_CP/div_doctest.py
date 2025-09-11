import numpy as np

def division(a: float, b: float):
    """Return a divided by b

    Args
    ----
    a : float
        Dividend. 
    b : float
        Divisor.

    Returns
    -------
    result : float
        Quotient.

    Examples, as doctests
    ---------------------
    >>> division(10, 5)
    TODO FIXME
    >>> division(10, 4)
    TODO FIXME
    >>> division(10, 0)
    TODO FIXME
    """
    return a / b


def division_vector(a_N: np.ndarray, b_N: np.ndarray):
    """Return vector a divided by vector b, with element-wise division

    Args
    ----
    a_N : np.ndarray
        1D array-like of dividends 
    b_N : np.ndarray, same shape as a
        1D array-like of divisors

    Returns
    -------
    result : np.ndarray
        1D array of element-wise quotient values

    Examples, as doctests
    ---------------------
    >>> unused = np.seterr(all='raise') # Make numerical warnings raise errors
    >>> division_vector(np.array([10, 10, 10]), np.array([5, 2, 1]))
    TODO FIXME
    >>> division_vector(np.array([10, 10, 10]), np.array([6, 4, 3]))
    TODO FIXME
    >>> division_vector(np.array([10, 10, 10]), np.array([2, 1, 0]))
    TODO FIXME
    """
    return a_N / b_N


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
