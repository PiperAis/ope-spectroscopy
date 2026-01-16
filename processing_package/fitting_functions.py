"""
V 1.0

Helper functions for fitting data with various mathematical functions.

Also includes functions to help identify peaks in spectra using
the second derivative test. Those functions work okay but could use
improvement.

Created March 18, 2025
Updated September 9, 2025

author: @piper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit
import xarray as xr


def quadratic(p, x):
    """ Generates values of a quadratic function of input x
    where p = [a, b, c] for the function ax^2 + bx + c"""
    x = np.array(x) 
    return p[0] * x**2 + p[1]*x + p[2]

def gaussian(x, *params):
    """ Generates a gaussian function of input x where params is a 
    list of lists: [amplitude, center, width] for each Gaussian """
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        amp = params[3 * i]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        y += amp * np.exp(-((x - cen) ** 2) / (2 * wid ** 2))
    return y

def Lorentz(p : list, x, N=None):
    """ p = [height, center, FWHM] """
    if N is None:
        N = len(p) // 3
    result = 0
    for i in range(N):
        area, center, FWHM = p[i*3], p[i*3+1], p[i*3+2]
        result += area / (1 + ((x - center) / (FWHM / 2))**2)
    return result

def exp_decay(t, A, B, C):
    return A * np.exp(-B * t) + C

def bi_exp_decay(t, A1, B1, A2, B2, C):
    return A1 * np.exp(-B1 * t) + A2 * np.exp(-B2 * t) + C

def diffL(p : list, x, y, N=None):
    ''' Takes the difference between y and a Lorentzian with 
    parameters p where p = [height, center, FWHM]'''
    difference = np.sum((y - Lorentz(p, x, N))**2)
    return difference

def diffquad(p, x, y):
    return np.sum((y-quadratic(p,x))**2)

def getDerivative(energies, intensity, smoothingWindow = 4):
    from scipy.signal import savgol_filter as smooth
    from numpy import gradient
    
    intensity = smooth(intensity, smoothingWindow, 3)
    
    dy = gradient(intensity)
    dx = gradient(energies)
    derivative = dy/dx
    derivative = smooth(derivative, smoothingWindow, 3)
    
    return derivative
    
def getDerivative2(energies, intensity, smoothingWindow = 4):
    from numpy import gradient
    
    dy = gradient(intensity)
    dx = gradient(energies)
    derivative = dy/dx
    derivative = smooth(derivative, smoothingWindow, 3)
    
    d2x = gradient(dx)
    d2y = gradient(dy)
    derivative2 = d2y/d2x
    derivative2 = smooth(derivative2, smoothingWindow, 3)
    
    return derivative2

def findPeaks(energies, intensity, smoothingWindow = 4, span = 5, title = ''):
    intensity = smooth(intensity, smoothingWindow, 3)
    
    derivative = getDerivative(energies, intensity, smoothingWindow)
    derivative2 = getDerivative2(energies, intensity, smoothingWindow)
    
    # Find indices where the derivative changes sign
    zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    
    filtered_zero_crossings = []
    for index in zero_crossings:
        if index >= span and index + span < len(derivative):
            if all(derivative[index - i] * derivative[index + i] < 0 for i in range(1, span + 1)):
                filtered_zero_crossings.append(index)
                
    zero_crossings = filtered_zero_crossings
    
    # Interpolate to find the actual zero crossing points
    x_zero_crossings = []
    
    for index in zero_crossings:
        x1, x2 = energies[index], energies[index + 1]
        y1, y2 = derivative[index], derivative[index + 1]
        # Linear interpolation formula
        x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
        x_zero_crossings.append(x_zero)
    
    # Show a figure to verify correct identification of maxima
    plt.figure(dpi=400)
    plt.plot(energies, intensity)
    # plt.plot(energies, derivative)
    plt.title(str(title))
    # Use 2nd derivative test to identify maxima
    peakEnergies = []
    for zero in zero_crossings:
        if derivative2[zero] < 0:
            peakEnergies.append(energies[zero])
            plt.axvline(energies[zero], color = 'gray', ls = "--")
    plt.show()
    
    return peakEnergies

def generateParams(inputpeaks : list, 
                   inputamps : list, 
                   boundsize : float = 0.003, 
                   wid : float = 0.001, 
                   widbound : list[float] = [0.001, 0.015]):
    """
    For fitting spectra.
    Generates a list of initial fitting parameters for gaussian or
    Lorentzian curves for each energy given in inputpeaks. 
    """
    p0 = []
    for i in range(len(inputpeaks)):
        p0.append(inputamps[i])
        p0.append(inputpeaks[i])
        p0.append(wid)

    bounds = []
    for i in range(len(inputpeaks)):
        bounds.append([0, None])
        bounds.append([inputpeaks[i]-boundsize, inputpeaks[i]+boundsize])
        bounds.append(widbound)
    
    return p0, bounds

def getFitParams(fit):
    fitpeaks = []
    fitamps = []
    fitwids = []
    for j in range(len(fit)//3):
        fitwids.append(fit[3*j+2])
        fitpeaks.append(round(fit[3*j+1], 4))
        fitamps.append(fit[3*j])
    return fitpeaks, fitamps, fitwids

# --- The below functions were written for looking at PL vs temperature data in summer 2024

def print_peaks(temperature, fitpeaks):
    print(temperature)
    for peak in fitpeaks:
        print(peak)

def print_amps(temperature, fitamps):
    print(temperature)
    for amp in fitamps:
        print(round(amp, 3))
        
def print_wids(temperature, fitwids):
    print(temperature)
    for wid in fitwids:
        print(round(wid, 4))

def get_spacing(number_list):
    spacings = []
    for i in range(len(number_list)):
        if i+1 < len(number_list):
            spacing = round(number_list[i+1]-number_list[i], 4)
            spacings.append(spacing)
            print(spacing)
    return spacings



