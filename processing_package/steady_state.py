"""
Holds helper functions for steady-state processing.

Rough draft, finishing touches will come during phase 5!
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import re
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.signal import savgol_filter as smooth
from . import fitting_functions as ff

xr.set_options(keep_attrs=True)

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

def diffG(p0: list, x, y, N=None):
    """
    p0: parameter vector [amplitude, center, FWHM, ...]
    Fixed parameters are handled via tight bounds in generateParams_fixed.
    """
    num_peaks = len(p0) // 3
    return np.sum((y - Gaussian(p0, x, num_peaks))**2)


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


def get_sample_name(filename: str) -> str:
    """
    Gets sample name from filename.
    Assumes pattern like 'flake1a-...'
    """
    match = re.search(r'flake.*-', filename)
    if match:
        sample_name = match.group(0)
        sample_name = sample_name.split('-')[0]
        sample_name = sample_name.replace('flake', '')
        return sample_name
    else:
        return 'unnamed sample'

def get_spot(filename: str) -> str:
    """
    Gets spot from filename.
    Assumes pattern like '-spot1-' after sample.
    """
    spot_match = re.search(r'spot.*-', filename)
    if spot_match:
        spot = spot_match.group(0)
        spot = spot.replace('-', '').replace('spot', 'Spot ')
        return spot
    return 'unknown spot'

def load_pl_data(filepath: Path) -> xr.DataArray:
    """
    Loads PL CSV data into xarray DataArray.
    Assumes columns: Wavelength, Intensity.
    Converts wavelength to energy in eV.
    """
    data = pd.read_csv(filepath, delimiter=',')
    wavelength = data['Wavelength'].values
    wavelength = wavelength
    energy = 1239.8 / wavelength # type: ignore
    intensity = data['Intensity'].values

    array = xr.DataArray(
        data=intensity,
        coords={'energy': energy},
        attrs={
            'filename': filepath.name,
            'sample': get_sample_name(filepath.name),
            'spot': get_spot(filepath.name)
        },
        name='intensity'
    )
    return array

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

def process_pl_directory_gaussian(experiment_directory) -> list:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    results = []

    for file in experiment_directory.iterdir():
        if file.is_file():
            try:
                data_array = load_pl_data(file)
                # maskcondition = (data_array.energy > 1.25) & (data_array.energy < 1.4)
                # da_masked = data_array.where(maskcondition, drop=True)
                fit_result = fit_gaussian(data_array,
                                          inputpeaks=[1.352, 1.334],
                                          inputamps=[400, 80],
                                          inputwids=[0.01, 0.025],
                                          fixed_energies={},
                                          fixed_widths={})
                result = {
                    'sample': f'{data_array.attrs['sample']} {data_array.attrs['spot']}',
                    'peak1_energy': fit_result['peak_energies'][0] if fit_result['peak_energies'] else None,
                    'peak1_width': fit_result['peak_widths'][0] if fit_result['peak_widths'] else None,
                    'peak1_area' : fit_result['peak_areas'][0] if fit_result['peak_areas'] else None,
                    'peak2_energy': fit_result['peak_energies'][1] if fit_result['peak_energies'] else None,
                    'peak2_width': fit_result['peak_widths'][1] if fit_result['peak_widths'] else None,
                    'peak2_area' : fit_result['peak_areas'][1] if fit_result['peak_areas'] else None,
                    'area2/area1' : round(float(fit_result['peak_areas'][1]/fit_result['peak_areas'][0]), 3)
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

    return results