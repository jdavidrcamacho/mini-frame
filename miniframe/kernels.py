# -*- coding: utf-8 -*-
import numpy as np


class kernel(object):
    """ Definition the base kernel class """
    is_kernel = True

    def __init__(self, *args):
        """ Puts all kernel arguments in an array pars """
        self.pars = np.array(args)

    def __call__(self, r):
        """ r = t - t' """
        raise NotImplementedError

    def __add__(self, b):
        if not hasattr(b, "is_kernel"):
            return Sum(Constant(c=float(b)), self)
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)


    def __mul__(self, b):
        if not hasattr(b, "is_kernel"):
            return Product(Constant(c=float(b)), self)
        return Product(self, b)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __repr__(self):
        """ Representation of each kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))


class _operator(kernel):
    """ To allow operations between two kernels """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ Sum of two kernels """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Product(_operator):
    """ Product of two kernels """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) * self.k2(r)


class Constant(kernel):
    """ This kernel returns its constant argument c """
    def __init__(self, c):
        super(Constant, self).__init__(c)
        self.c = c

    def __call__(self, r):
        return self.c * np.ones_like(r)


# Squared exponential kernel
class SquaredExponential(kernel):
    """ 
    Squared Exponential kernel, also known as radial basis function 
    (RBF kernel) in other works. Equation 25 in the paper.

    Parameters:
        ell = length-scale, lambda in the paper
    """
    def __init__(self, ell):
        super(SquaredExponential, self).__init__(ell)
        self.ell = ell
    
    def __call__(self, r):
        f1 = r**2
        f2 = self.ell**2
        return np.exp(-0.5 * f1/f2)


class dSE_dt1(SquaredExponential):
    """ 
    Derivative of the SquaredExponential kernel with respect to t1.
    Equation A4 in the paper, for N=1.
    """
    def __init__(self, ell):
        super(dSE_dt1, self).__init__(ell)
        self.ell = ell
    
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell**2 
        return -f1/f2 * np.exp(-0.5* f1*f1/f2)


class dSE_dt2(SquaredExponential):
    """
    Derivative of the SquaredExponential kernel with respect to t1.
    Equation A5 in the paper, for N=1.
    """
    def __init__(self, ell):
        super(dSE_dt2, self).__init__(ell)
        self.ell = ell
    
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell**2
        return f1/f2 * np.exp(-0.5* f1*f1/f2)
        


class ddSE_dt2dt1(SquaredExponential):
    """
    Second derivative of the SquaredExponential kernel with respect to t1 and t2.
    Equation A6 in the paper, for N=1.
    """
    def __init__(self, ell):
        super(ddSE_dt2dt1, self).__init__(ell)
        self.ell = ell

    def __call__(self, r):
        f1 = r**2
        f2 = self.ell**2
        return (1/f2 - f1/f2**2) * np.exp(-0.5* f1/f2)


# Quasi-periodic kernel
class QuasiPeriodic(kernel):
    """ 
    This kernel is the product between the exponential sine squared kernel 
    and the squared exponential kernel. It is known as the quasi-periodic kernel.
    Equation 27 in the paper.

    Parameters:
        ell_p = length scale of the periodic component
        ell_e = evolutionary time scale
        period
    """
    def __init__(self, ell_p, ell_e, period):
        super(QuasiPeriodic, self).__init__(ell_p, ell_e, period)
        self.ell_p = ell_p
        self.ell_e = ell_e
        self.period = period

    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell_p**2
        f3 = self.ell_e**2
        f4 = self.period
        
        f5 = np.sin(np.pi*f1/f4)
        return np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )


class dQP_dt1(QuasiPeriodic):
    """ 
    Derivative of the QuasiPeriodic kernel with respect to t1.
    Equation A8 in the paper.
    """
    def __init__(self, ell_p, ell_e, period):
        super(dQP_dt1, self).__init__(ell_p, ell_e, period)
        self.ell_p = ell_p
        self.ell_e = ell_e
        self.period = period

    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell_p**2
        f3 = self.ell_e**2
        f4 = self.period

        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        return (-(4*np.pi*f5*f6)/(f2*f4) - f1/f3) *f7 


class dQP_dt2(QuasiPeriodic):
    """ 
    Derivative of the QuasiPeriodic kernel with respect to t1.
    Equation A9 in the paper.
    """
    def __init__(self, ell_p, ell_e, period):
        super(dQP_dt2, self).__init__(ell_p, ell_e, period)
        self.ell_p = ell_p
        self.ell_e = ell_e
        self.period = period

    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell_p**2
        f3 = self.ell_e**2
        f4 = self.period

        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        return ((4*np.pi*f5*f6)/(f2*f4) + f1/f3) *f7 


class ddQP_dt2dt1(QuasiPeriodic):
    """
    Second derivative of the QuasiPeriodic kernel with respect to t1 and t2.
    Equation A10 in the paper.
    """
    def __init__(self, ell_p, ell_e, period):
        super(ddQP_dt2dt1, self).__init__(ell_p, ell_e, period)
        self.ell_p = ell_p
        self.ell_e = ell_e
        self.period = period

    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.ell_p**2
        f3 = self.ell_e**2
        f4 = self.period
        
        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        f8 = (-(4*np.pi*f5*f6)/(f2*f4) - f1/f3)
        f9 = ((4*np.pi*f5*f6)/(f2*f4) + f1/f3) 
        return f8*f9*f7 + (1.0/f3 + 4*np.pi*np.pi*f6*f6/(f2*f4*f4) /
                           -4*np.pi*np.pi*f5*f5/(f2*f4*f4)) *f7 

