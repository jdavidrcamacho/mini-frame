# -*- coding: utf-8 -*-
import kernels
import numpy as np

#def entry_parameters(a):
#    """
#        entry_parameters() assigns the entry parameters to their
#    respectives kernels
#    
#        Parameters
#    a = array with the important parameters
#    
#        Returns
#    an array with the kernels
#    """
#    l,vc,vr,lc,bc,br = a
#    return np.array([kernels.SquaredExponential(l),
#                     kernels.dSE_dt1(l),
#                     kernels.dSE_dt2(l),
#                     kernels.ddSE_dt2dt1(l),
#                     kernels.Const1(vc),
#                     kernels.Const2(vr),
#                     kernels.Const3(lc),
#                     kernels.Const4(bc),
#                     kernels.Const5(br)])

def build_smallmatrix(kernel, x):
    """
        build_smallmatrix() creates the smaller covariance matrices,
    equations 18 to 23 in the paper.
        
        Parameters
    kernel = kernel in use
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    K = covariance matrix
    """ 
    r = x[:, None] - x[None, :]
    K = kernel(r)

    return K


def k11(a,x):
    """
        Equation 18
    """
    l,vc,vr,lc,bc,br = a
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x) 
    gammadgdg = build_smallmatrix(kernels.ddSE_dt2dt1(l),x)  
    gammagdg = build_smallmatrix(kernels.dSE_dt1(l),x)  
    gammadgg = build_smallmatrix(kernels.dSE_dt2(l),x)  

    return vc**2 * gammagg + vr**2* gammadgdg + vc*vr*(gammagdg + gammadgg)
    
def k22(a,x):
    """
        Equation 19
    """
    l,vc,vr,lc,bc,br = a
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x)
    
    return lc**2 * gammagg

def k33(a,x):
    """
        Equation 20
    """
    l,vc,vr,lc,bc,br = a  
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x) 
    gammadgdg = build_smallmatrix(kernels.ddSE_dt2dt1(l),x)  
    gammagdg = build_smallmatrix(kernels.dSE_dt1(l),x)  
    gammadgg = build_smallmatrix(kernels.dSE_dt2(l),x)  
    
    return bc**2 * gammagg + br**2* gammadgdg + bc*br*(gammagdg + gammadgg)

def k12(a,x):
    """
        Equation 21
    """
    l,vc,vr,lc,bc,br = a
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x) 
    gammagdg = build_smallmatrix(kernels.dSE_dt1(l),x)  

    return vc*lc * gammagg + vr*lc*gammagdg

def k13(a,x):
    """
        Equation 22
    """
    l,vc,vr,lc,bc,br = a
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x) 
    gammadgdg = build_smallmatrix(kernels.ddSE_dt2dt1(l),x)  
    gammagdg = build_smallmatrix(kernels.dSE_dt1(l),x)  
    gammadgg = build_smallmatrix(kernels.dSE_dt2(l),x)  
    
    return vc*bc*gammagg + vr*br* gammadgdg + vc*br*gammagdg + vr*bc*gammadgg


def k23(a,x):
    """
        Equation 23
    """
    l,vc,vr,lc,bc,br = a
    
    gammagg  = build_smallmatrix(kernels.SquaredExponential(l),x) 
    gammagdg = build_smallmatrix(kernels.dSE_dt1(l),x)  

    return bc*lc * gammagg + br*lc*gammagdg

def build_bigmatrix(a,x,y,yerr):
    """
        build_bigmatrix() creates the big covariance matrix,
    equations 24 in the paper.
        
        Parameters
    a = array with the important parameters
    x = range of values of the independent variable (usually time)
    y = range of values of te dependent variable (the measurments)
    yerr = error in the measurments
    
        Returns
    K = covariance matrix
    """ 
    K11 = k11(a,x)
    K22 = k22(a,x)
    K33 = k33(a,x)
    K12 = k12(a,x)
    K13 = k13(a,x)
    K23 = k23(a,x)
    
    K1 = np.hstack((K11,K12,K13))
    K2 = np.hstack((K12,K22,K23))
    K3 = np.hstack((K13,K23,K33))
    
    K = np.vstack((K1,K2,K3))
    K = K + yerr**2*np.identity(len(yerr))

    return K

