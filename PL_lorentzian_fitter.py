"""
PL Peak Fitter Script

Processes photoluminescence CSV files from a directory, extracts metadata from filenames,
loads data into xarray DataArrays, fits Lorentzians, and extracts peak energies.

Similar to TRRDataset approach but simplified for PL data.

Author: Adapted from TRRDataset and directory_quickplot.py
"""
#%%
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import re
import sys
from os.path import dirname
import matplotlib.pyplot as plt
from scipy.optimize import minimize

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import config
from processing_package import fitting_functions as ff

xr.set_options(keep_attrs=True)

def Lorentz(p : list, x, N=None):
    """ p = [height, center, FWHM] """
    if N is None:
        N = len(p) // 3
    result = 0
    for i in range(N):
        area, center, FWHM = p[i*3], p[i*3+1], p[i*3+2]
        result += area / (1 + ((x - center) / (FWHM / 2))**2)
    return result

def diffL_fixed(p0: list, x, y, N=None):
    """
    p0: parameter vector with fixed peak centers removed.
    fixed_energies: dict of {peak_index: fixed_energy}, e.g. {1: 1.325}
    Rebuilds full parameter vector before evaluating Lorentz.
    """
    num_peaks = len(p0) // 3
    
    return np.sum((y - Lorentz(p0, x, num_peaks))**2)


def generateParams_fixed(inputpeaks, inputamps, inputwids,
                         fixed_energies: dict = {},
                         fixed_widths: dict = {0 : 0.02},
                         boundsize=0.003,
                         widbound=[0.001, 0.05]):
    """
    fixed_energies: dict of {peak_index: fixed_energy}
    Excludes fixed peak centers from p0 and bounds.
    """
    p0, bounds = [], []
    for i in range(len(inputpeaks)):
        p0.append(inputamps[i])
        bounds.append([0, None])
        if i not in fixed_energies.keys():          # only add energy as free param if not fixed
            p0.append(inputpeaks[i])
            bounds.append([inputpeaks[i] - boundsize, inputpeaks[i] + boundsize])
        else:
            p0.append(fixed_energies[i])
            bounds.append([fixed_energies[i], fixed_energies[i]])
        if i not in fixed_widths.keys():          # only add energy as free param if not fixed
            p0.append(inputwids[i])
            bounds.append(widbound)
        else:
            p0.append(fixed_widths[i])
            bounds.append([fixed_widths[i], fixed_widths[i]])

    return p0, bounds


def get_sample_name(filename: str) -> str:
    """
    Gets sample name from filename, similar to TRRDataset.
    Assumes pattern like 'flake1a-...'
    """
    match = re.search(r'flake.*-', filename)
    if match:
        sample_name = match.group(0)
        sample_name = sample_name.split('-')[0]
        sample_name = sample_name.replace('flake', 'Flake ')
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
    Adapted from directory_quickplot.py read_spectrum.
    Assumes columns: Wavelength, Intensity.
    Converts to energy.
    """
    data = pd.read_csv(filepath, delimiter=',')
    wavelength = data['Wavelength'].values
    energy = 1239.8 / wavelength  # type: ignore
    intensity = data['Intensity'].values

    # Create DataArray
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

def fit_lorentzian(data_array, inputpeaks, inputamps, inputwids,
                   fixed_energies: dict = {}, fixed_widths: dict = {}) -> dict:  # e.g. fixed_energies={1: 1.325}
    x = data_array.energy.values
    y = data_array.values
    num_peaks = len(inputpeaks)

    p0, bounds = generateParams_fixed(inputpeaks, inputamps, inputwids,
                                      fixed_energies=fixed_energies, fixed_widths = fixed_widths)

    try:
        fit_free = minimize(diffL_fixed, p0,
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
            lor_j = ff.Lorentz([amp_j, cen_j, wid_j], x)
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

def process_pl_directory(experiment_date: str) -> list:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    directory = config.PL_DIR / 'raw' / experiment_date
    results = []

    for file in directory.iterdir():
        if file.is_file():
            try:
                data_array = load_pl_data(file)
                fit_result = fit_lorentzian(data_array,
                             inputpeaks=[1.351, 1.327],
                             inputamps=[1000, 200],
                             inputwids=[0.025, 0.05],
                             fixed_energies={},
                             fixed_widths = {0 : 0.0125})
                result = {
                    'filename': file.name,
                    'sample': data_array.attrs['sample'],
                    'spot': data_array.attrs['spot'],
                    'peak1_energy': fit_result['peak_energies'][0] if fit_result['peak_energies'] else None,
                    'peak1_amplitude': fit_result['peak_amplitudes'][0] if fit_result['peak_amplitudes'] else None,
                    'peak2_energy': fit_result['peak_energies'][1] if fit_result['peak_energies'] else None,
                    'peak2_amplitude': fit_result['peak_amplitudes'][1] if fit_result['peak_amplitudes'] else None,
                    'fit_success': fit_result['fit_success']
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

    return results

if __name__ == "__main__":
    # Example usage
    experiment_date = '2026-03-03'
    results = process_pl_directory(experiment_date)

    # Print results
    for res in results:
        print(res)


# %%
