from __future__ import print_function
from numpy import exp, log
import numpy as np
from scipy.special import gammaln

try:
    from itertools import imap
except ImportError:
    imap=map

def lfact(x):
    '''Compute the log factorial using the scipy gammaln function.

    This is commonly referred to as Stirlings approximation/formula for factorials.'''
    return gammaln(x+1)

def nanexp(x):
    '''Computes the exponential of x, and replaces nan and inf with finite numbers.

    Returns an array or scalar replacing Not a Number (NaN) with zero, (positive) infinity with a very large number and negative infinity with a very small (or negative) number.'''
    return np.nan_to_num(np.exp(x))
