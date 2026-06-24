"""
Helper functions for processing PL vs. magnetic field (B-sweep) data.

Ported from pipersfunctions.py v1.3.
"""
from pathlib import Path
from . import TRR_functions
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.constants as const

def get_file_paths(
                    dir : Path, 
                    condition : str = ''
                    ) -> list[Path]:
    """
    Returns a list of the file names in dir in alphabetical order.

    :param dir: Directory containing files to be listed
    :type dir: pathlib.PurePath
    :param condition: Conditional for filtering file types.
    [f for f in dir.iterdir() if condition]
    :type condition: str
    """
    if condition:
        filepaths = [f for f in dir.iterdir() if condition] #type: ignore
    else:
        filepaths = [f for f in dir.iterdir()] #type: ignore

    return filepaths

def bfield_contour_plot(data_directory : Path):
    
    bfield_file_list = get_file_paths(data_directory)
    spectra = []
    for filepath in sorted(bfield_file_list):
        df = pd.read_csv(filepath)                    # one spectrum
        field = TRR_functions.get_field(filepath)          # reuse the existing helper
        da = xr.DataArray(
            df['Intensity'].values,
            coords={"wavelength": df['Wavelength'].values,
                    },
            dims=["wavelength"],
        )
        energy_values = const.h * const.c / (da['wavelength'].values * 1e-9) / const.e
        da = da.expand_dims(field=[field])              # tag this spectrum with its field
        da = da.assign_coords(
            energy=("wavelength", energy_values)
            )
        spectra.append(da)

    bfield_map = xr.concat(spectra, dim="field")       # stack into 2D
    bfield_map.plot.contourf(x="field", y="energy")    # contour map, axes auto-labeled
    plt.show()
###

# import glob
# import os
# import re

# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import savgol_filter as smooth


# def norm(x):
#     return (x - np.min(x)) / (np.max(x) - np.min(x))


# def read_spectrum(
#         filename: str, normalize: bool = False, xEnergy: bool = True,
#         delimiter=',', umBlaze: bool = False):
#     """
#     Reads data from a txt file or CSV, including data collected using
#     LightField. umBlaze applies a 22 nm shift to compensate for a known
#     calibration error with one of the MRI lab gratings.

#     Returns x-axis (wavelength or energy), intensity, and max intensity.
#     """
#     data = np.genfromtxt(filename, delimiter=delimiter, skip_header=1).T
#     wavelength = data[0]

#     if umBlaze:
#         wavelength -= 22

#     energy = 1239.8 / wavelength
#     intensity = data[1]
#     maxIntensity = max(intensity)

#     if normalize:
#         intensity = norm(intensity)

#     if xEnergy:
#         return wavelength, intensity, maxIntensity
#     else:
#         return energy, intensity, maxIntensity


# def mask(xdata, ydata, xmin, xmax):
#     """Returns masked x and y arrays, keeping only points where xmin <= x <= xmax."""
#     indices = [i for i, x in enumerate(xdata) if xmin <= x <= xmax]
#     return xdata[indices], ydata[indices]





# def get_heatmap_data(energy, intensity,
#                      plot_with_mask: bool = False,
#                      apply_smooth: bool = False,
#                      min_energy=None,
#                      max_energy=None):
#     """Prepares x and z data for a heatmap, optionally masking and/or smoothing."""
#     if plot_with_mask:
#         if min_energy is None or max_energy is None:
#             raise ValueError(
#                 "min_energy and max_energy must be specified when plot_with_mask is True."
#             )
#         yvar, intensity = mask(energy, intensity, min_energy, max_energy)
#     else:
#         yvar = energy
#     zvar = intensity
#     if apply_smooth:
#         zvar = smooth(zvar, 5, 3)
#     return yvar, zvar


# def plot_heatmap(
#         xvariable: list,
#         yvariable: list,
#         zvariable: list,
#         plot_title: str,
#         cmap: str = 'inferno',
#         colorbar: bool = False) -> None:
#     """
#     Plots a contourf heatmap of z as a function of x (horizontal) and y (vertical).

#     xvariable: 1-D list of x values (e.g. B-fields)
#     yvariable: list of equal-length y-axis arrays, one per x value
#     zvariable: list of intensity arrays, one per x value
#     """
#     xvariable_array = np.array(xvariable)
#     yvariable_array = np.array(yvariable)
#     zvariable_array = np.array(zvariable)

#     x_grid, y_grid = np.meshgrid(xvariable_array, yvariable_array[0])

#     plt.figure(dpi=400, figsize=(4, 4))
#     contour = plt.contourf(x_grid, y_grid, zvariable_array.T, cmap=cmap)
#     if colorbar:
#         plt.colorbar(contour, label='Intensity')

#     plt.xlabel(r'$\mu_0H$ (T)', size=16)
#     plt.ylabel('Energy (eV)', size=16)
#     if plot_title:
#         plt.title(plot_title, size=16)
#     plt.show()
