import numpy as np

class MeanModel(object):
	""" A base class for GP mean functions """
	def __init__(self, arg):
		self.arg = arg


class Constant(MeanModel):
	""" A constant offset mean function """
	def __init__(self, c):
		super(Constant, self).__init__(c)

	def __call__(self, t):
		""" Evaluate this mean function at times t """
		t = np.atleast_1d(t)
		return c*np.ones_like(t)


class Linear(MeanModel):
	""" 
	A linear mean function
	m(t) = slope * t + intercept 
	"""
	def __init__(self, slope, intercept):
		super(Linear, self).__init__(slope, intercept)

	def __call__(self, t):
		""" Evaluate this mean function at times t """
		t = np.atleast_1d(t)
		return slope * t + intercept


class Parabola(MeanModel):
	""" 
	A 2nd degree polynomial mean function
	m(t) = quad * t**2 + slope * t + intercept 
	"""
	def __init__(self, quad, slope, intercept):
		super(Parabola, self).__init__(quad, slope, intercept)

	def __call__(self, t):
		""" Evaluate this mean function at times t """
		t = np.atleast_1d(t)
		return slope * t + intercept