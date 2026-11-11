"""
Helper functions for fitting data with various mathematical functions.

Also includes functions to help identify peaks in spectra using
the second derivative test. Those functions work okay but could use
improvement, or may be replaceable by an existing third-party package.

There might be some useful ways to bring native xr operations in here, 
since I intend to use xr objects for all data analysis.

author: @piper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth
import pandas as pd
from scipy.optimize import curve_fit
import xarray as xr
from . import utilities

#---Mathematical Functions---#

def quadratic(p, x):
    """ Generates values of a quadratic function of input x
    where p = [a, b, c] for the function ax^2 + bx + c"""
    x = np.array(x) 
    return p[0] * x**2 + p[1]*x + p[2]

def Gaussian(p: list, x, N=None):
    """ p = [amplitude, center, FWHM, amplitude, center, FWHM, ...] """
    if N is None:
        N = len(p) // 3
    result = 0
    for i in range(N):
        amp, center, FWHM = p[i*3], p[i*3+1], p[i*3+2]
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        result += amp * np.exp(-((x - center)**2) / (2 * sigma**2))
    return result

def Lorentz(p : list, x, N=None):
    """ p = [height, center, FWHM] """
    if N is None:
        N = len(p) // 3
    result = 0
    for i in range(N):
        area, center, FWHM = p[i*3], p[i*3+1], p[i*3+2]
        result += area / (1 + ((x - center) / (FWHM / 2))**2)
    return result

def peak_area(lineshape: str, amp: float, fwhm: float) -> float:
    """Integrated area of one peak ('gaussian' or 'lorentzian') from its height and FWHM."""
    if lineshape == 'gaussian':
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amp * sigma * np.sqrt(2 * np.pi)
    if lineshape == 'lorentzian':
        return amp * np.pi * fwhm / 2
    raise ValueError(f"Unknown lineshape: {lineshape!r}")

def exp_decay(t, A, B, C):
    """
    A (result[0]): Amplitude
    B (result[1]): 1/(time constant)
    C (result[2]): Offset
    """
    return A * np.exp(-B * t) + C

def param_covariance(jac, resid) -> np.ndarray:
    """
    Parameter covariance from a least-squares solution: cov = (JᵀJ)⁺ · SSE/(N − p),
    i.e. residual variance estimated from the fit itself (as scipy's curve_fit does).
    Columns of J are normalized before inverting: parameters on very different
    scales (e.g. heights ~1e3 vs centers ~1) otherwise make JᵀJ look singular and the
    pseudo-inverse silently drops real (large) uncertainty along correlated directions.
    Returns NaNs if there are no degrees of freedom.
    """
    jac = np.asarray(jac, dtype=float)
    n, p = jac.shape
    if n <= p:
        return np.full((p, p), np.nan)
    s2 = float(np.sum(np.asarray(resid) ** 2)) / (n - p)
    d = np.linalg.norm(jac, axis=0)
    d[d == 0] = 1.0
    js = jac / d
    return np.linalg.pinv(js.T @ js) / np.outer(d, d) * s2


def fit_quality(resid, y, n_params: int) -> dict:
    """
    Goodness of fit from a least-squares solution's residuals and the fitted data.
    No per-point σ is available, so this is RMSE (√(SSE/dof)) and R², not χ².
    Returns n_points, dof, rmse and r_squared (NaN when dof ≤ 0 or y is constant).
    """
    y = np.asarray(y, dtype=float)
    n_points = len(y)
    dof = n_points - n_params
    sse = float(np.sum(np.asarray(resid) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(sse / dof)) if dof > 0 else np.nan
    r_squared = 1 - sse / sst if dof > 0 and sst > 0 else np.nan
    return {'n_points': n_points, 'dof': dof, 'rmse': rmse, 'r_squared': r_squared}

# lineshape name -> model function
LINESHAPES = {
    'gaussian':   Gaussian,
    'lorentzian': Lorentz,
}

def diffquad(p, x, y):
    return np.sum((y-quadratic(p,x))**2)

#----Generating and accessing fit parameters----#

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


def generateParams_fixed(inputpeaks, inputamps, inputwids,
                         fixed_energies: dict = {},
                         fixed_widths: dict = {},
                         boundsize=0.008,
                         widbound=[0.001, 0.05]):
    """
    Generates initial parameters and bounds for a multi-peak fit (used by PLFitter).
    fixed_energies: dict of {peak_index: fixed_energy}
    fixed_widths: dict of {peak_index: fixed_width}
    Fixes parameters via tight bounds rather than removing from parameter vector.
    """
    p0, bounds = [], []
    for i in range(len(inputpeaks)):
        p0.append(inputamps[i])
        bounds.append([0, None])
        if i not in fixed_energies.keys():
            p0.append(inputpeaks[i])
            bounds.append([inputpeaks[i] - boundsize, inputpeaks[i] + boundsize])
        else:
            p0.append(fixed_energies[i])
            bounds.append([fixed_energies[i], fixed_energies[i]])
        if i not in fixed_widths.keys():
            p0.append(inputwids[i])
            bounds.append(widbound)
        else:
            p0.append(fixed_widths[i])
            bounds.append([fixed_widths[i], fixed_widths[i]])

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


#----Old stuff to fix up or get rid of----#

# def findPeaks(energies, intensity, smoothingWindow = 4, span = 5, title = ''):
#     intensity = smooth(intensity, smoothingWindow, 3)
    
#     derivative = getDerivative(energies, intensity, smoothingWindow)
#     derivative2 = getDerivative2(energies, intensity, smoothingWindow)
    
#     # Find indices where the derivative changes sign
#     zero_crossings = np.where(np.diff(np.sign(derivative)))[0]
    
#     filtered_zero_crossings = []
#     for index in zero_crossings:
#         if index >= span and index + span < len(derivative):
#             if all(derivative[index - i] * derivative[index + i] < 0 for i in range(1, span + 1)):
#                 filtered_zero_crossings.append(index)
                
#     zero_crossings = filtered_zero_crossings
    
#     # Interpolate to find the actual zero crossing points
#     x_zero_crossings = []
    
#     for index in zero_crossings:
#         x1, x2 = energies[index], energies[index + 1]
#         y1, y2 = derivative[index], derivative[index + 1]
#         # Linear interpolation formula
#         x_zero = x1 - y1 * (x2 - x1) / (y2 - y1)
#         x_zero_crossings.append(x_zero)
    
#     # Show a figure to verify correct identification of maxima
#     plt.figure(dpi=400)
#     plt.plot(energies, intensity)
#     # plt.plot(energies, derivative)
#     plt.title(str(title))
#     # Use 2nd derivative test to identify maxima
#     peakEnergies = []
#     for zero in zero_crossings:
#         if derivative2[zero] < 0:
#             peakEnergies.append(energies[zero])
#             plt.axvline(energies[zero], color = 'gray', ls = "--")
#     plt.show()
    
#     return peakEnergies

# --- The below functions were written for looking at PL vs temperature data in summer 2024
# Probably not needed anymore but holding onto them a bit longer until we've implemented
# alternatives.

# def print_peaks(temperature, fitpeaks):
#     print(temperature)
#     for peak in fitpeaks:
#         print(peak)

# def print_amps(temperature, fitamps):
#     print(temperature)
#     for amp in fitamps:
#         print(round(amp, 3))
        
# def print_wids(temperature, fitwids):
#     print(temperature)
#     for wid in fitwids:
#         print(round(wid, 4))

# def get_spacing(number_list):
#     spacings = []
#     for i in range(len(number_list)):
#         if i+1 < len(number_list):
#             spacing = round(number_list[i+1]-number_list[i], 4)
#             spacings.append(spacing)
#             print(spacing)
#     return spacings



