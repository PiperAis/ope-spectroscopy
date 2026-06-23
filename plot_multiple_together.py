"""

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
from scipy.signal import savgol_filter as smooth

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import config

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

def process_pl_directory(experiment_date: str) -> None:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    directory = config.PL_DIR / 'raw' / experiment_date
    results = []

    plt.figure(figsize=(6, 6), dpi=400)
    labellist = ['$MoS_2$ HS', '$WSe_2$ HS', 'Bare CrSBr']
    i = 0
    for file in directory.iterdir():
        if file.is_file():
            data_array = load_pl_data(file)
            maskcondition = (data_array.energy > 1.26) & (data_array.energy < 1.38)
            da_masked = data_array.where(maskcondition, drop=True)
            # x = data_array.energy.values
            # y = data_array.values/data_array.values.max()
            x = da_masked.energy.values
            y = da_masked.values #/data_array.values.max()
            y = smooth(y, 15, 3)
            plt.plot(x, y, label=f'{labellist[i]}')
            i += 1
    plt.legend()
    plt.xlabel('Energy (eV)', fontsize = 'large')
    plt.ylabel('PL Intensity (arb. units)', fontsize = 'x-large')
    plt.show()

if __name__ == "__main__":
    experiment_date = '2026-03-03'
    process_pl_directory(experiment_date)
#%%