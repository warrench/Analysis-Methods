# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:44:16 2019

@author: warrenc
"""

import numpy as np
from scipy.optimize import curve_fit

def rotate_data(data,theta):
    """
    Rotate the complex data by a fixed angle to put all the information into a
    single quadrature
    
    Parameters:
        data:  Numpy array containing the complex IQ data
        theta: Rotation angle in degrees
        
    Returns:
        data_r(i)_rot: The rotated data
    """
    data = np.exp(1j*np.pi*theta/180.0)*data
    return np.real(data), np.imag(data)

def data_stats(data, axis=None, calcerr=False):
    """
    Calculate the average and standard error across the array
    
    Parameters:
        data:    Numpy array containing the individual measurement traces to
                 be averaged over
        axis:    The averaging axis. Default is to average across the flattened
                 array
        calcerr: Check whether to return the standard error of the traces
        
    Returns:
        data_avg: An array of the averaged traces
        data_err: An array of the standard errors of the traces
              
    """
    data_avg = np.mean(data, axis=axis)
    data_std = np.std(data, axis=axis)
    if calcerr:
        return data_avg, data_std/np.sqrt(data.shape[axis])
    else:
        return data_avg
    
def reshape_inter(data,K,n_C):
    """
    Reshape the 2-indexed array of Logger data for the interleaved randomized
    benchmarking data into a 3D array of data for each Clifford gate in the set
    
    Parameters:
        data: 2-indexed Numpy array containing the RB data
        K:    Number of sequences for each Clifford gate to be averaged over
        n_C:  Number of Clifford gates in the set for interleaved RB
        
    Returns:
        data_inter: 3-indexed Numpy array of size (n_C,K,m)
    """
    m = data.shape[1]
    data_inter = np.reshape(data,(n_C,K,m))
    return data_inter

def rb_fitzero(m,p,A,B):
    """
    Fit to the zeroth order model of the sequence fidelity
    """
    return A*p**m + B

def rb_fitfirst(m,p,A,B,C):
    """
    Fit to the first order model of the sequence fidelity
    """
    return A*p**m + B*(m-1)*p**(m-2) + C

def calc_fit(m,data, order=0):
    """
    Calculate the fit parameters for the sequence fidelity
    
    Parameters:
        m:           x-array of the number of clifford gates
        data:        The averaged RB data
        order:       which fit order to use
        
    Returns:
        popt: A list containing the fit parameters to the model the first of
              which is p
    """
    if order == 1:
        popt, pcov = curve_fit(rb_fitfirst,m,data)
    else:
        popt, pcov = curve_fit(rb_fitzero,m,data)
    return popt

def rb_error(p,d):
    """
    Calculate the randomized benchmarking average gate error over all gates
    
    Parameters:
        p: The depolarizing parameter from the fit
        d: Dimension of the gate
        
    Returns:
        r: The error rate over all Clifford gates
    """
    return (1.0-1.0/d)*(1.0-p)

def rb_gateerror(p,p_C,d):
    """
    Calculate the individual gate error from interleaved RB
    
    Parameters:
        p:   Depolarizing parameter from regular RB
        p_C: Depolarizing parameter from interleaved RB fit
        d:   Dimension of the gate
        
    Returns:
        r_C: The error rate for the individual Clifford gate
        
    """
    return (1.0-1.0/d)*(1.0-p_C/p)