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

        Parameters
    SE_l = lamba in the paper   
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
    equation 16 in the paper.
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
        f1 = r
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

#	def dSE_dt1(self, r):
#		"""
#			Derivative in order to t1, 
#		equation 16 in the paper.
#		Important: not to forget that d/dt2 = -d/dt1 
#		"""
#        f1 = r**2
#        f2 = self.SE_l**2 
#        return f1/(f2**2) * np.exp(-0.5* f1/f2)
#
#	def dSE_dt2(self, r):
#		"""
#			Derivative in order to t2, 
#		equation A5 in the paper.
#		Important: not to forget that d/dt2 = -d/dt1 
#		"""
#        f1 = r**2
#        f2 = self.SE_l**2 
#        return -f1/(f2**2) * np.exp(-0.5* f1/f2)
#		
#	def ddSE_dt2dt1(self, r):
#		"""
#			Derivative in order to t2 and t1, 
#		equation 17 in the paper.
#		"""
#        f1 = r**2
#        f2 = self.SE_l**2
#        return (1/f2**2 - f1/f2**4) * np.exp(-0.5* f1/f2)

	
class Const1(kernel):
    """
        Constant Vc in the paper.
    """
    def __init__(self, Vc):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(Const1, self).__init__(Vc)
        self.Vc = Vc
    	
    def __call__(self, r):
    	"""
    		Not a kernel but we are defining it similarly,
    	not sure if it is the smartest move.
    	"""                 
        return self.Vc
     
       
class Const2(kernel):
    """
        Constant Vr in the paper.
    """
    def __init__(self, Vr):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(Const2, self).__init__(Vr)
        self.Vr = Vr
    	
    def __call__(self, r):
    	"""
    		Not a kernel but we are defining it similarly,
    	not sure if it is the smartest move.
    	"""                 
        return self.Vr
     
       
class Const3(kernel):
    """
        Constant Vr in the paper.
    """
    def __init__(self, Lc):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(Const3, self).__init__(Lc)
        self.Lc = Lc
    	
    def __call__(self, r):
    	"""
    		Not a kernel but we are defining it similarly,
    	not sure if it is the smartest move.
    	"""                 
        return self.Lc
        
       
class Const4(kernel):
    """
        Constant Bc in the paper.
    """
    def __init__(self, Bc):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(Const4, self).__init__(Bc)
        self.Bc = Bc
    	
    def __call__(self, r):
    	"""
    		Not a kernel but we are defining it similarly,
    	not sure if it is the smartest move.
    	"""                 
        return self.Bc
        
       
class Const5(kernel):
    """
        Constant Vr in the paper.
    """
    def __init__(self, Br):
        """
        	Because we are "overwriting" the function __init__
        we use this weird super function.
        """
        super(Const5, self).__init__(Br)
        self.Br = Br
    	
    def __call__(self, r):
    	"""
    		Not a kernel but we are defining it similarly,
    	not sure if it is the smartest move.
    	"""                 
        return self.Br
        
