# -*- coding: utf-8 -*-
"""
Using autograd for the kernel derivatives.
This is slower than the hand-coded version in kernels.py
but arguably much easier to read.
The API is changed 
- from having derivatives as childs of each main kernel
- to having methods `d_t1`, `d_t2`, `dd_t`, etc for each main kernel
Not sure which one is best.
"""

import autograd.numpy as np
from autograd import elementwise_grad as egrad

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

    return0 = lambda self, r: np.zeros_like(r)
    d_t1 = return0
    d_t2 = return0
    dd_t = return0
    ddd_t1 = return0
    ddd_t2 = return0
    dddd_t = return0

class SquaredExponential(kernel):
    """ 
    Squared Exponential kernel, also known as radial basis function 
    (RBF kernel) in other works.

    Parameters:
        ell = length-scale, lambda in the paper
    """
    def __init__(self, ell):
        super(SquaredExponential, self).__init__(ell)
        self.ell = ell
    
    def __call__(self, r):
        return np.exp( -0.5 * (r**2)/(self.ell**2) )

    def d_t1(self, r):
        """ derivative with respect to t1 (or t, or ti in the paper) """
        return egrad(self.__call__)(r)

    def _d_t1_check(self, r):
        """ hand-coded derivative to check """
        f1 = r
        f2 = self.ell**2 
        return -f1/f2 *np.exp(-0.5*f1*f1/f2)

    def d_t2(self, r):
        """ derivative with respect to t2 (or t', or tj in the paper) """
        return -egrad(self.__call__)(r)

    def _d_t2_check(self, r):
        """ hand-coded derivative to check """
        f1 = r
        f2 = self.ell**2
        return f1/f2 *np.exp(-0.5*f1*f1/f2)


    # now comes the magic...

    def dd_t(self, r):
        """ second derivative with respect to t1 and t2 """
        # same as ddSE_dt2dt1
        return egrad(self.d_t2)(r)

    def ddd_t1(self, r):
        """ third derivative with respect to t1 """
        # same as dddSE_dt2ddt1
        return egrad(self.dd_t)(r)

    def ddd_t2(self, r):
        """ third derivative with respect to t2 """
        # same as dddSE_ddt2dt1
        return -egrad(self.dd_t)(r)

    def dddd_t(self, r):
        """ fourth derivative with respect to t1 and t2 """
        # same as ddddSE_ddt2ddt1
        return egrad(self.ddd_t2)(r)




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
        s = np.sin(np.pi*r/self.period)**2
        return np.exp( - 2*s/(self.ell_p**2) - (r*r)/(2*self.ell_e**2))


    def d_t1(self, r):
        """ derivative with respect to t1 (or t, or ti in the paper) """
        # same as dQP_dt1
        return egrad(self.__call__)(r)

    def d_t2(self, r):
        """ derivative with respect to t2 (or t', or tj in the paper) """
        # same as dQP_dt2
        return -egrad(self.__call__)(r)


    def dd_t(self, r):
        """ second derivative with respect to t1 and t2 """
        # same as ddQP_dt2dt1
        return egrad(self.d_t2)(r)

    def ddd_t1(self, r):
        """ third derivative with respect to t1 """
        # same as dddQP_dt2ddt1
        return egrad(self.dd_t)(r)

    def ddd_t2(self, r):
        """ third derivative with respect to t2 """
        # same as dddQP_ddt2dt1
        return -egrad(self.dd_t)(r)

    def dddd_t(self, r):
        """ fourth derivative with respect to t1 and t2 """
        # same as ddddQP_ddt2ddt1
        return egrad(self.ddd_t2)(r)

