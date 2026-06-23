"""
Helper functions for processing PL vs. magnetic field (B-sweep) data.

Ported from pipersfunctions.py v1.3.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter as smooth


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def read_spectrum(
        filename: str, normalize: bool = False, xEnergy: bool = True,
        delimiter=',', umBlaze: bool = False):
    """
    Reads data from a txt file or CSV, including data collected using
    LightField. umBlaze applies a 22 nm shift to compensate for a known
    calibration error with one of the MRI lab gratings.

    Returns x-axis (wavelength or energy), intensity, and max intensity.
    """
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=1).T
    wavelength = data[0]

    if umBlaze:
        wavelength -= 22

    energy = 1239.8 / wavelength
    intensity = data[1]
    maxIntensity = max(intensity)

    if normalize:
        intensity = norm(intensity)

    if xEnergy:
        return wavelength, intensity, maxIntensity
    else:
        return energy, intensity, maxIntensity


def mask(xdata, ydata, xmin, xmax):
    """Returns masked x and y arrays, keeping only points where xmin <= x <= xmax."""
    indices = [i for i, x in enumerate(xdata) if xmin <= x <= xmax]
    return xdata[indices], ydata[indices]


def extract_field_from_filename(filename: str):
    """
    Extracts B-field value string from a filename expecting the format e.g. '1p5T'.
    Raises ValueError if no field is found.
    """
    parts = filename.split('-')
    for part in parts:
        match = re.search(r"(.*?)T", part)
        if match:
            return match.group(1)
    raise ValueError(
        f"No magnetic field found in filename {filename}. "
        "Ensure the filename contains a field value followed by 'T' (e.g. '0p5T')."
    )


def get_b_fields(filepath_list: list):
    """
    Extracts and sorts B-field values from a list of file paths.

    Returns:
        bfields: sorted list of float field values
        bfield_list: sorted list of (bfield, filepath) tuples
    """
    bfields = []
    bfield_list = []
    for filepath in filepath_list:
        filename = os.path.basename(filepath)
        bfield_str = extract_field_from_filename(filename)
        if bfield_str is None:
            print(f"No magnetic field found in filename {filename}. Skipping.")
            continue
        if "p" in bfield_str:
            bfield_str = bfield_str.replace("p", ".")
        if 'n' in bfield_str:
            bfield_str = bfield_str.replace("n", "-")
        bfield = float(bfield_str)
        bfields.append(bfield)
        bfield_list.append((bfield, filepath))
    bfield_list.sort(key=lambda x: x[0])
    bfields.sort()
    return bfields, bfield_list


def get_heatmap_data(energy, intensity,
                     plot_with_mask: bool = False,
                     apply_smooth: bool = False,
                     min_energy=None,
                     max_energy=None):
    """Prepares x and z data for a heatmap, optionally masking and/or smoothing."""
    if plot_with_mask:
        if min_energy is None or max_energy is None:
            raise ValueError(
                "min_energy and max_energy must be specified when plot_with_mask is True."
            )
        yvar, intensity = mask(energy, intensity, min_energy, max_energy)
    else:
        yvar = energy
    zvar = intensity
    if apply_smooth:
        zvar = smooth(zvar, 5, 3)
    return yvar, zvar


def plot_heatmap(
        xvariable: list,
        yvariable: list,
        zvariable: list,
        plot_title: str,
        cmap: str = 'inferno',
        colorbar: bool = False) -> None:
    """
    Plots a contourf heatmap of z as a function of x (horizontal) and y (vertical).

    xvariable: 1-D list of x values (e.g. B-fields)
    yvariable: list of equal-length y-axis arrays, one per x value
    zvariable: list of intensity arrays, one per x value
    """
    xvariable_array = np.array(xvariable)
    yvariable_array = np.array(yvariable)
    zvariable_array = np.array(zvariable)

    x_grid, y_grid = np.meshgrid(xvariable_array, yvariable_array[0])

    plt.figure(dpi=400, figsize=(4, 4))
    contour = plt.contourf(x_grid, y_grid, zvariable_array.T, cmap=cmap)
    if colorbar:
        plt.colorbar(contour, label='Intensity')

    plt.xlabel(r'$\mu_0H$ (T)', size=16)
    plt.ylabel('Energy (eV)', size=16)
    if plot_title:
        plt.title(plot_title, size=16)
    plt.show()
