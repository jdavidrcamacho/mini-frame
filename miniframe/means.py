import numpy as np

from inspect import signature
from functools import wraps
 
def auto_args(f):
    sig = signature(f)  # Get a signature object for the target:
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        f(self, *args, **kwargs)
    return replacement


def array_input(f):
    @wraps(f)
    def wrapped(self, t):
        t = np.atleast_1d(t)
        r = f(self, t)
        return r
    return wrapped


# class MeanModelTemplate(object):
#   """ Mean functions should follow this strucuture """
#   @auto_args
#   def __init__(self, par1, par2, par3, ...)
#       pass
#   def __call__(self, t):
#       # Evaluate this mean function at array of times t


class Constant(object):
    """ A constant offset mean function """
    @auto_args
    def __init__(self, c):
        pass

    @array_input
    def __call__(self, t):
        return self.c * np.ones_like(t)


class Linear(object):
    """ 
    A linear mean function
    m(t) = slope * t + intercept 
    """
    @auto_args
    def __init__(self, slope, intercept):
        pass

    @array_input
    def __call__(self, t):
        return self.slope * t + self.intercept


class Parabola(object):
    """ 
    A 2nd degree polynomial mean function
    m(t) = quad * t**2 + slope * t + intercept 
    """
    @auto_args
    def __init__(self, quad, slope, intercept):
        pass

    @array_input
    def __call__(self, t):
        return self.quad * t**2 + self.slope * t + self.intercept


class Polynomial(object):
    """ 
    An N degree polynomial mean function
    m(t) = coeff[0]*t**(N-1) + coeff[1]*t**(N-2) + ... + coeff[N-2]*t + coeff[N-1]
    """
    @auto_args
    def __init__(self, coeff):
        self.degree = len(coeff) - 1

    @array_input
    def __call__(self, t):
        return np.polyval(self.coeff, t)



class Keplerian(object):
    """
    A Keplerian mean function
    Parameters are the orbital period P, semi-amplitude K, eccentricity e,
    argument of the periastron w, and time of periastron T0
    """
    @auto_args
    def __init__(self, P, K, e, w, T0):
        pass

    @array_input(self, t):
        pass