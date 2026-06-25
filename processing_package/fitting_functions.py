"""
V 2.0

Helper functions for fitting data with various mathematical functions.

Also includes functions to help identify peaks in spectra using
the second derivative test. Those functions work okay but could use
improvement.

Created March 18, 2025
Updated September 9, 2025
Updated June 24, 2026

author: @piper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth
from scipy.optimize import minimize
import pandas as pd
from scipy.optimize import curve_fit, minimize, differential_evolution
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

def exp_decay(t, A, B, C):
    return A * np.exp(-B * t) + C

def bi_exp_decay(t, A1, B1, A2, B2, C):
    return A1 * np.exp(-B1 * t) + A2 * np.exp(-B2 * t) + C

#----Differences between data and fits, could be collapsed into one function later----#

def diffG(p0: list, x, y, N=None):
    """
    p0: parameter vector [amplitude, center, FWHM, ...]
    Fixed parameters are handled via tight bounds in generateParams_fixed.
    """
    num_peaks = len(p0) // 3
    return np.sum((y - Gaussian(p0, x, num_peaks))**2)

def diffL(p : list, x, y, N=None):
    ''' Takes the difference between y and a Lorentzian with 
    parameters p where p = [height, center, FWHM]'''
    difference = np.sum((y - Lorentz(p, x, N))**2)
    return difference

def diffquad(p, x, y):
    return np.sum((y-quadratic(p,x))**2)

#----Generating parameters----#

def generateParams_fixed(inputpeaks, inputamps, inputwids,
                         fixed_energies: dict = {},
                         fixed_widths: dict = {},
                         boundsize=0.008,
                         widbound=[0.001, 0.05]):
    """
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


#----Fitting----#

def fit_gaussian(data_array, inputpeaks, inputamps, inputwids,
                 fixed_energies: dict = {}, fixed_widths: dict = {},
                 method = 'global') -> dict:
    """
    Fits N Gaussians to the data and extracts peak energies.
    fixed_energies: dict of {peak_index: fixed_energy}, e.g. {1: 1.325}
    fixed_widths: dict of {peak_index: fixed_width}, e.g. {0: 0.0125}
    """
    x = data_array.energy.values
    y = data_array.values
    # y -= y.min()
    y = smooth(y, 10, 3)
    y -= min(y)
    num_peaks = len(inputpeaks)

    p0, bounds = generateParams_fixed(inputpeaks, inputamps, inputwids,
                                      fixed_energies=fixed_energies,
                                      fixed_widths=fixed_widths)

    try:
        if method == 'global':
            globalbounds = [(b[0], y.max() if b[1] is None else b[1]) for b in bounds]
            # globalbounds = [(200, 1.5*y.max()), (1.4, 1.55), (0.001, 0.05), (200, y.max()), (1.32, 1.335), (0.001, 0.05)]
            results = differential_evolution(diffG, globalbounds,
                                args=(x, y),
                                maxiter=1000, tol=1e-2,
                                polish=True)
            print(f"DE success: {results.success}, message: {results.message}")
            fit = results.x
        else:    
            fit = minimize(diffG, p0,
                        args=(x, y),
                        bounds=bounds).get('x')

        peak_energies = [float(round(fit[3*j + 1], 4)) for j in range(num_peaks)]
        peak_amplitudes = [float(round(fit[3*j], 1)) for j in range(num_peaks)]
        peak_widths = [float(round(fit[3*j + 2], 4)) for j in range(num_peaks)]
        fit_success = True

        maskcondition = (data_array.energy > 1.25) & (data_array.energy < 1.4)
        da_masked = data_array.where(maskcondition, drop=True)
        x = da_masked.energy.values
        y = da_masked.values
        y = smooth(y, 15, 3)
        
        # Plot
        plt.figure(figsize=(6, 6), dpi = 400)
        plt.plot(x, y, label='Data', color='black')
        plt.plot(x, Gaussian(fit, x), label='Fit', color='red')
        peak_areas = []
        for j in range(num_peaks):
            amp_j = fit[3*j]
            cen_j = fit[3*j + 1]
            wid_j = fit[3*j + 2]
            area_j = (amp_j * wid_j) / (2 * np.pi)
            peak_areas.append(float(round(area_j, 3)))
            gauss_j = Gaussian([amp_j, cen_j, wid_j], x)
            label = f'Peak {j+1}: {cen_j:.4f} eV'
            if j in fixed_energies.keys():
                label += ' (fixed E)'
            if j in fixed_widths.keys():
                label += ' (fixed W)'
            plt.plot(x, gauss_j, label=label, linestyle='--')
        plt.legend()
        plt.title(f"{data_array.attrs['sample']} {data_array.attrs['spot']}")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (arb. units)')
        plt.show()

    except Exception as e:
        print(f"Fit failed for {data_array.attrs['filename']}: {e}")
        peak_energies = None
        peak_amplitudes = None
        num_peaks = 0
        fit_success = False

    return {
        'peak_energies': peak_energies,
        'peak_amplitudes': peak_amplitudes,
        'peak_areas' : peak_areas,
        'peak_widths' : peak_widths,
        'num_peaks': num_peaks,
        'fit_success': fit_success,
    }

def fit_lorentzian(data_array, inputpeaks, inputamps, inputwids,
                   fixed_energies: dict = {}, fixed_widths: dict = {}) -> dict:  # e.g. fixed_energies={1: 1.325}
    x = data_array.energy.values
    y = data_array.values
    num_peaks = len(inputpeaks)

    p0, bounds = generateParams_fixed(inputpeaks, inputamps, inputwids,
                                      fixed_energies=fixed_energies, fixed_widths = fixed_widths)

    try:
        fit_free = minimize(diffL, p0,
                            args=(x, y, fixed_energies),
                            bounds=bounds).get('x')

        # Rest of your plotting/extraction code stays identical, using p_full
        peak_energies = [float(round(fit_free[3*j + 1], 4)) for j in range(num_peaks)]
        peak_amplitudes = [float(round(fit_free[3*j], 4)) for j in range(num_peaks)]
        fit_success = True

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label='Data', color='black')
        plt.plot(x, Lorentz(fit_free, x), label='Fit', color='red')
        # Individual peaks
        for j in range(num_peaks):
            amp_j = fit_free[3*j]
            cen_j = fit_free[3*j + 1]
            wid_j = fit_free[3*j + 2]
            lor_j = Lorentz([amp_j, cen_j, wid_j], x)
            plt.plot(x, lor_j, label=f'Peak {j+1}: {cen_j:.3f} eV', linestyle='--')
        plt.legend()
        plt.title(f"{data_array.attrs['sample']} {data_array.attrs['spot']} ({num_peaks} peaks)")
        plt.xlabel('Energy (eV)')
        plt.ylabel('Intensity (arb. units)')
        plt.show()

    except Exception as e:
        print(f"Fit failed for {data_array.attrs['filename']}: {e}")
        peak_energies = None
        num_peaks = 0
        fit_success = False
        params = None

    return {
        'peak_energies': peak_energies,
        'peak_amplitudes' : peak_amplitudes,
        'num_peaks': num_peaks,
        'fit_success': fit_success,
    }

#----Old stuff to fix up or get rid of----#

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



