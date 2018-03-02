# -*- coding: utf-8 -*-
import numpy as np


class kernel(object):
    """ 
        Definition the kernel class and its properties, 
    for more informations check the paper of Rajpaul et al., 2015. 
    """
    def __init__(self, *args):
        """
        	Puts all kernel arguments in an array pars.
        """
        self.pars = np.array(args)

    def __call__(self, r):
    	"""
    		r = t - t'
    	"""
        raise NotImplementedError
        #return self.k1(x1, x2, i, j) * self.k2(x1, x2, i, j)

    def __add__(self, b):
        return Sum(self, b)
    def __radd__(self, b):
        return self.__add__(b)
    def __mul__(self, b):
        return Product(self, b)
    def __rmul__(self, b):
        return self.__mul__(b)

    def __repr__(self):
        """
        	Representation of each kernel instance.
        """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

                            
class _operator(kernel):
    """
    	To allow operations between two kernels.
    """
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """
    	To allow the sum of kernels.
    """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Product(_operator):
    """ 
    	To allow the product of kernel.
    """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, r):
        return self.k1(r) * self.k2(r)
        

class SquaredExponential(kernel):
    """
        Squared Exponential kernel and derivatives,
    also know as radial basis function (RBF kernel) in other works.
        Equation 25 in the paper.

        Parameters
    SE_l = lambda in the paper   
    """
    def __init__(self, SE_l):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(SquaredExponential, self).__init__(SE_l)
        self.SE_l = SE_l
	
    def __call__(self, r):
    	"""
    		Squared exponential kernel,
    	equation 25 in the paper.
    	"""
        f1 = r**2 
        f2 = self.SE_l**2                   
        return np.exp(-0.5* f1/f2)	

class dSE_dt1(SquaredExponential):
    """
        Derivative in order to t1, 
    equation 16 and A4 in the paper.
    """
    def __init__(self, SE_l):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(dSE_dt1, self).__init__(SE_l)
        self.SE_l = SE_l
	
    def __call__(self, r):
        f1 = r
        f2 = self.SE_l**2 
        return -f1/f2 * np.exp(-0.5* f1*f1/f2)

class dSE_dt2(SquaredExponential):
    """
        Derivative in order to t2, 
    equation A5 in the paper.
    """
    def __init__(self, SE_l):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(dSE_dt2, self).__init__(SE_l)
        self.SE_l = SE_l
	
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.SE_l**2
        return f1/f2 * np.exp(-0.5* f1*f1/f2)
		
class ddSE_dt2dt1(SquaredExponential):
    """
        Derivative in order to t2 and t1, 
    equation 17 in the paper.
    """
    def __init__(self, SE_l):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(ddSE_dt2dt1, self).__init__(SE_l)
        self.SE_l = SE_l
	
    def __call__(self, r):
        f1 = r**2
        f2 = self.SE_l**2
        return (1/f2 - f1/f2**2) * np.exp(-0.5* f1/f2)

   
class QuasiPeriodic(kernel):
    """
        Definition of the product between the exponential sine squared kernel 
    and the squared exponential  kernel, also known as quasi periodic kernel.
        Equation 27 in the paper.

        Parameters
    QP_lp = length scale of the periodic component
    QP_le = evolutionary time scale
    QP_P = period
    """
    def __init__(self, QP_lp, QP_le, QP_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(QuasiPeriodic, self).__init__(QP_lp, QP_le, QP_P)
        self.QP_lp = QP_lp
        self.QP_le = QP_le
        self.QP_P = QP_P    

    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.QP_lp**2
        f3 = self.QP_le**2
        f4 = self.QP_P
        
        f5 = np.sin(np.pi*f1/f4)
        return np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )

class dQP_dt1(QuasiPeriodic):
    """
        Derivative in order to t1, 
    equation 16 and A4 in the paper.
    """
    def __init__(self, QP_lp, QP_le, QP_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(dQP_dt1, self).__init__(QP_lp, QP_le, QP_P)
        self.QP_lp = QP_lp
        self.QP_le = QP_le
        self.QP_P = QP_P   
	
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.QP_lp**2
        f3 = self.QP_le**2
        f4 = self.QP_P
        
        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        return (-(4*np.pi*f5*f6)/(f2*f4) - f1/f3) *f7 

class dQP_dt2(QuasiPeriodic):
    """
        Derivative in order to t2, 
    equation A5 in the paper.
    """
    def __init__(self, QP_lp, QP_le, QP_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(dQP_dt2, self).__init__(QP_lp, QP_le, QP_P)
        self.QP_lp = QP_lp
        self.QP_le = QP_le
        self.QP_P = QP_P   
	
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.QP_lp**2
        f3 = self.QP_le**2
        f4 = self.QP_P
        
        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        return ((4*np.pi*f5*f6)/(f2*f4) + f1/f3) *f7 

class ddQP_dt2dt1(QuasiPeriodic):
    """
        Derivative in order to t2 and t1, 
    equation A10 in the paper.
    """
    def __init__(self, QP_lp, QP_le, QP_P):
        """
        Because we are "overwriting" the function __init__
        we use this weird super function
        """
        super(ddQP_dt2dt1, self).__init__(QP_lp, QP_le, QP_P)
        self.QP_lp = QP_lp
        self.QP_le = QP_le
        self.QP_P = QP_P   
	
    def __call__(self, r):
        f1 = np.abs(r)
        f2 = self.QP_lp**2
        f3 = self.QP_le**2
        f4 = self.QP_P
        
        f5 = np.sin(np.pi*f1/f4)
        f6 = np.cos(np.pi*f1/f4)
        f7 = np.exp( -(2.0*f5*f5/f2) - 0.5*f1/f3 )
        f8 = (-(4*np.pi*f5*f6)/(f2*f4) - f1/f3)
        f9 = ((4*np.pi*f5*f6)/(f2*f4) + f1/f3) 
        return f8*f9*f7 + (1.0/f3 + 4*np.pi*np.pi*f6*f6/(f2*f4*f4) /
                           -4*np.pi*np.pi*f5*f5/(f2*f4*f4)) *f7 

