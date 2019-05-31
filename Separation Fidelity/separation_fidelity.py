# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:53:56 2019

@author: warrenc
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapz

def histogram_data(data, bins=50, range=None):
    """
    Takes in an array of and produces an array of counts for the histogram of
    the size as the specified number of bins. If range is None it will auto 
    determine the size of the bins. By specifying a range you bin within an
    interval specified by range=(min,max).
    
    This method also outputs an array containing the center of the bins to
    integrate along so as to normalize the data
    
    Inputs:
        data: 1D array of data
        bins: The total number of bins to sort counts into
        range: A tuple (bmin,bmax) containing the bounds to form bins over 
        
    Returns:
        hist: A 1-D array contianing the counts of the histogram
        center: A 1-D array containing the center of the bins of the data
        norm: The normalization of the histogram
    """
    
    hist, histbins = np.histogram(data, bins=bins, range=range)
    center = (histbins[:-1] + histbins[1:])/2
    norm = trapz(hist,x=center)
    
    return hist, center, norm

def gaussian(x, A, mu, std):
    return A*np.exp(-(x-mu)**2/(2.0*std**2))

def double_gauss(x, A1, mu1, std1, A2, mu2, std2):
    return gaussian(x,A1,mu1,std1) + gaussian(x,A2,mu2,std2)

def fit_histogram(xdata,ydata, bounds=None):
    """
    Fit the histogram to a double Gaussian 
    """
    # bounds = ([A1_min, mu1_min, std1_min, A2_min, mu2_min, std2_min],
    #           [A1_max, mu1_max, std1_max, A2_max, mu2_max, std2_max])

    if bounds is None:
        xmin = min(xdata)
        xmax = max(xdata)
        ymax = max(ydata)
        bounds = ([0, xmin, 0, 0, xmin, 0],
                  [1.1*ymax, xmax, xmax-xmin, 1.1*ymax, xmax, xmax-xmin])
    
    popt,pcov = curve_fit(double_gauss,xdata,ydata,bounds)
    return popt

def find_intersections(hist1,hist2):
    """
    hist1 and hist2 must have the same size
    """
    return np.argwhere(np.diff(np.sign(hist1-hist2))).flatten()

def calculation_separation_fidelity(centerbins,hist1,hist2):
    """
    hist1 should always be oriented such that its main central Gaussian peak is
    always oriented to the left of hist2's main peak
    """
    # Find the intersection of the histograms
    idx = find_intersections(hist1,hist2)
    idx = idx[int(len(list)/2)]
    # Normalize the histograms (may already be normalized)
    norm1 = trapz(hist1,x=centerbins)
    norm2 = trapz(hist2,x=centerbins)
    hist1 = hist1/norm1
    hist2 = hist2/norm2
    # Integrate the area under the gaussian corresponding to each side of the
    # separation axis
    err_1 = trapz(hist1[idx:])
    err_2 = trapz(hist2[:idx+1])
    # Return the separation fidelity
    return 1.0-err_1-err_2