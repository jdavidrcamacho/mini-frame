import numpy as np
from functools import wraps

__all__ = ['Constant', 'Linear', 'Parabola', 'Keplerian']


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


class MeanModel(object):
    _parsize = 0
    def __init__(self, *pars):
        self.pars = list(pars)

    def __repr__(self):
        """ Representation of each instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))

    @classmethod
    def initialize(cls):
        return cls( *([0.]*cls._parsize) )


class Constant(MeanModel):
    """ A constant offset mean function """
    _parsize = 1
    def __init__(self, c):
        super(Constant, self).__init__(c)

    @array_input
    def __call__(self, t):
        return self.pars[0] * np.ones_like(t)


class Linear(MeanModel):
    """ 
    A linear mean function
    m(t) = slope * t + intercept 
    """
    _parsize = 2
    def __init__(self, slope, intercept):
        super(Linear, self).__init__(slope, intercept)

    @array_input
    def __call__(self, t):
        return self.pars[0] * t + self.pars[1]


class Parabola(object):
    """ 
    A 2nd degree polynomial mean function
    m(t) = quad * t**2 + slope * t + intercept 
    """
    _parsize = 2
    def __init__(self, quad, slope, intercept):
        pass

    @array_input
    def __call__(self, t):
        return self.pars[0] * t**2 + self.pars[1] * t + self.pars[2]


class Keplerian(MeanModel):
    """
    Keplerian function
    tan[phi(t) / 2 ] = sqrt(1+e / 1-e) * tan[E(t) / 2] = true anomaly
    E(t) - e*sin[E(t)] = M(t) = eccentric anomaly
    M(t) = (2*pi*t/tau) + M0 = Mean anomaly
    P  = period in days
    e = eccentricity
    K = RV amplitude in m/s 
    w = longitude of the periastron
    T0 = zero phase
    
    RV = K[cos(w+v) + e*cos(w)] + sis_vel
    """
    _parsize = 5
    def __init__(self, P, K, e, w, T0):
        super(Keplerian, self).__init__(P, K, e, w, T0)

    @array_input
    def __call__(self, t):
        P, K, e, w, T0 = self.pars
        #mean anomaly
        Mean_anom=[2*np.pi*(x1-T0)/P  for x1 in t]
        #eccentric anomaly -> E0=M + e*sin(M) + 0.5*(e**2)*sin(2*M)
        E0=[x1 + e*np.sin(x1)  + 0.5*(e**2)*np.sin(2*x1) \
                                                            for x1 in Mean_anom]
        #mean anomaly -> M0=E0 - e*sin(E0)
        M0=[x1 - e*np.sin(x1) for x1 in E0]

        i=0
        while i<100:
            #[x + y for x, y in zip(first, second)]
            calc_aux=[x2-y2 for x2,y2 in zip(Mean_anom,M0)]
            E1=[x3 + y3/(1-e*np.cos(x3)) for x3,y3 in zip(E0,calc_aux)]
            M1=[x4 - e*np.sin(x4) for x4 in E0]
            i+=1
            E0=E1
            M0=M1
        nu = [2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(x5/2)) for x5 in E0]
        RV = [K*(e*np.cos(w)+np.cos(w+x6)) for x6 in nu]
        return RV
