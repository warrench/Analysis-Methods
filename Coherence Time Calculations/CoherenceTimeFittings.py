# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:48:57 2019

@author: warrenc

The idea of this script is to take all of the data traces and fit their decay
times. The information of the decay is split between the I and Q. To remedy
this we want to rotate all of the information into one of the quadratures for
each trace and then fit to the maximized quadrature. The first process is to
find the angle which minimizes the spread of data in one of the quadratures.
We minimize the Q quadrature over some angle theta to push all the information
into I by minimizing the std(Q) and maximizing the std(I)
"""

import numpy as np
from scipy.optimize import minimize, curve_fit

def opt_angle(theta, data):
    """
    This function is passed to the scipy minimization method to find the
    angle which minimizes the std in the Q quadrature. This corresponds to
    a rotation angle which aligns the data with the I axis.
    
    Parameters:
        theta: The rotation angle
        data:  IQ data
        
    Returns:
        A cost function to be minimized. In this situation we simply minimize
        the standard deviation of the Q-axis        
    """
    
    data_rot = data*np.exp(1.0j*theta*np.pi/180.0)
    Q_rot = np.imag(data_rot)
    #I_rot = np.real(data_rot)
    return np.std(Q_rot)#+1.0/np.std(I_rot)

def rotate2I(data,theta0=45.0,returntheta=False):
    """
    Rotate all of the data into the I quadrature. This function takes in the
    raw data (I + iQ) and finds the optimal angle with which to rotate to
    minimize the std of Q corresponding to aligning the data along the I axis.
    This returns the rotated axis and if chosen the optimized angle.
    
    
    Parameters:
        data:        A 1D IQ trace
        theta0:      An initial guess to the optimal angle
        returntheta: Return the computed optimal angle
    
    Returns:
        data_rot:  The rotated data
        theta_opt: The optimized rotation angle
    """
    res = minimize(opt_angle,theta0,args=data)
    theta_opt = res.x
    data_rot = data*np.exp(1.0j*theta_opt*np.pi/180.0)
    if returntheta:
        return data_rot, theta_opt
    else:
        return data_rot

def T1_fit(t,A,T1,B):
    """
    Fitting function for T1
    """
    return A*np.exp(-t/T1)+B

def fit_all(t,data2D,bounds=None):
    """
    Take in all T1 traces from a Logger file and fit T1 from repeated T1
    measurements. Special care needs to be taken to rescale the x and y data
    for the curve_fit to fit properly. It tends to act somewhat wildly and
    bounds should always be included so that the fit parameters don't blow up.
    
    Parameters:
        t:      x-axis data (1-D array of times scaled appropriately)
        data2D: 2D array of T1 traces from sequential runs
        bounds: a tuple of bounds for the curve fitting algorithm of the form
                ([A_min,T1_min,B_min],[A_max,T1_max,B_max])
                
    Returns:
        fit_params: A nested list of the computed fit parameters for the T1s
                    [[A_0,T1_0,B_0],[A_1,T1_1,B_1],...]
    """
    fit_params = []
    for i in range(data2D.shape[0]):
        trace = data2D[i,:]
        y = np.real(rotate2I(trace))
        popt, pcov = curve_fit(T1_fit,t,y,bounds=bounds)
        fit_params.append(popt)
    return fit_params




