"""
Holds helper functions for steady-state processing.

Rough draft, finishing touches will come during phase 5!
"""

import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth
from . import fitting_functions as ff
from . import utilities

xr.set_options(keep_attrs=True)

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
            'sample': utilities.get_sample_name(filepath.name),
            'spot': utilities.get_spot(filepath.name)
        },
        name='intensity'
    )
    return array

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
                fit_result = ff.fit_gaussian(data_array,
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

def plot_multiple_together(directory: Path) -> None:
    """
    Plots all PL spectra in a directory on a single figure.
    Applies an energy mask (1.26–1.38 eV) and Savitzky-Golay smoothing.
    """
    labellist = ['$MoS_2$ HS', '$WSe_2$ HS', 'Bare CrSBr']
    plt.figure(figsize=(6, 6), dpi=400)
    i = 0
    for file in directory.iterdir():
        if file.is_file():
            data_array = load_pl_data(file)
            maskcondition = (data_array.energy > 1.26) & (data_array.energy < 1.38)
            da_masked = data_array.where(maskcondition, drop=True)
            x = da_masked.energy.values
            y = da_masked.values
            y = smooth(y, 15, 3)
            plt.plot(x, y, label=f'{labellist[i]}')
            i += 1
    plt.legend()
    plt.xlabel('Energy (eV)', fontsize='large')
    plt.ylabel('PL Intensity (arb. units)', fontsize='x-large')
    plt.show()


def process_pl_directory_lorentzian(experiment_directory) -> list:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    results = []

    for file in experiment_directory.iterdir():
        if file.is_file():
            try:
                data_array = load_pl_data(file)
                fit_result = ff.fit_lorentzian(data_array,
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