"""
v1.0 Updated 2026/03/06

Currently set up to generate plots of all datasets in a PL experiment file.

Change experiment_date variable to match the name of the folder containing your data.

To do:
- Implement saving
- Use xarray instead of pandas
"""
#%%
from os.path import dirname, isfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import yaml

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

# ---- User Input ---- #

experiment_date = '2026-03-02'

# Path to the project config file. Change if your config.yaml lives elsewhere.
config_path = Path(THIS_DIR) / 'config.yaml'


def load_config(config_path):
    """Load config.yaml and resolve any relative paths relative to the yaml file."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    for key in ('data_dir', 'processed_data_dir', 'reports_dir', 'pl_dir'):
        if key in cfg:
            cfg[key] = (base / cfg[key]).resolve()
    return cfg


cfg = load_config(config_path)


def norm(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def read_spectrum(
        filename : Path,
        normalize : bool = False,
        xEnergy : bool = True,
        delimiter : str = ',',
        umBlaze : bool = False):
    """
    Reads data from a txt file or CSV, including data collected using
    LightField. umBlaze = 1 sets a 22 nm shift, to compensate for a
    known calibration error with one of the gratings in the MRI lab.

    Returns 2 lists: x-axis (either WL or energy) and y-axis (intensity)
    Returns the max intensity of the (un-normalized) data.
    I can't remember why it does this.
    """

    data = pd.read_csv(file, delimiter = delimiter)
    wavelength = data['Wavelength'].values

    # Apply calibration adjustment
    if umBlaze:
        wavelength -= 22

    # Convert from wavelength to energy
    energy = 1239.8/wavelength

    intensity = data['Intensity'].values
    maxIntensity = intensity.max()

    if normalize:
        intensity = norm(intensity)

    if xEnergy:
        return wavelength, intensity, maxIntensity
    else:
        return energy, intensity, maxIntensity

def mask(xdata, ydata, xmin, xmax):
    """
    Returns 2 lists: masked x-var data and masked y-var data.
    """
    indicesWithinRange = [index for index, x in enumerate(xdata)
                          if xmin <= x <= xmax]
    xdataMasked = xdata[indicesWithinRange]
    ydataMasked = ydata[indicesWithinRange]
    return xdataMasked, ydataMasked

def quick_plot(title : str, x_var, y_var,
               x_label : str, y_label : str,
               highdef : bool = False) -> None:
    if highdef == True:
        plt.figure(dpi=400)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_var, y_var)
    plt.show()

# Plot all in directory

directory = cfg['pl_dir'] / 'raw' / experiment_date

for file in directory.iterdir():
    if isfile(file):
        filename = file.stem
        x, y, max = read_spectrum(file)
        # x, y = mask(x, y, 1, 2)
        quick_plot(title = f'{filename}', x_var = x, y_var = y, x_label = 'Energy (ev)', y_label = 'PL (arb units)')
# %%
