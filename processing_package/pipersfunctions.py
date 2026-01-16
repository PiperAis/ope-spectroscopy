"""
Functions for plotting and processing data

v 1.2
Updated September 9, 2025

Moved some functions to other modules.
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import re
import xarray as xr

from . import fitting_functions as ff

def norm(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def read_spectrum(filename : str, normalize : bool = False, xEnergy : bool = True, delimiter=',', umBlaze : bool = False):
    """ 
    Reads data from a txt file or CSV, including data collected using 
    LightField. umBlaze = 1 sets a 22 nm shift, to compensate for a 
    known calibration error with one of the gratings in the MRI lab. 

    Returns 2 lists: x-axis (either WL or energy) and y-axis (intensity)
    Returns the max intensity of the (un-normalized) data.
    I can't remember why it does this.
    """

    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=1).T
    wavelength = data[0]

    # Apply calibration adjustment
    if umBlaze:
        wavelength -= 22
    
    # Convert from wavelength to energy
    energy = 1239.8/wavelength

    intensity = data[1]
    maxIntensity = max(intensity)

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
    indicesWithinRange = [index for index, x in enumerate(xdata) if xmin <= x <= xmax]
    xdataMasked = xdata[indicesWithinRange]
    ydataMasked = ydata[indicesWithinRange]
    return xdataMasked, ydataMasked

def extract_field_from_filename(filename : str):
    """ 
    Old version which expects bfield in file name as (e.g.) 1.5T 
    """
    parts = filename.split('-')
    for part in parts:
        # match = re.search(r"(?:\_|-)?(.*?)T", part)
        match = re.search(r"(.*?)T", part)
        if match:
            return match.group(1)
    raise ValueError(f"No magnetic field found in filename {filename}. Please ensure the filename contains a magnetic field value followed by 'T' (e.g., '0.5T').")


def get_b_fields(filepathList: list):
    """ 
    Written for generation of contour maps of PL vs bfield.
    """
    bfields = []
    bfieldList = []
    for filepath in filepathList:
        filename = os.path.basename(filepath)
        bfield_str = extract_field_from_filename(filename)
        if bfield_str is None:
            print(f"Warning: No magnetic field found in filename {filename}. Skipping...")
            continue
        if "p" in bfield_str:
            bfield_str = bfield_str.replace("p", ".")
        if 'n' in bfield_str:
            bfield_str = bfield_str.replace("n", "-")
        bfield = float(bfield_str)
        bfields.append(bfield)
        bfieldList.append((bfield, filepath))
    bfieldList.sort(key=lambda x: x[0])
    bfields.sort()
    return bfields, bfieldList

def extract_temperature_from_filename(filename):
    """
    Old version which expects temp in the form (e.g.) 2.2K
    """
    match = re.search(r"(?:\_)?(.*?)K", filename)
    if match:
        return int(match.group(1))
    else:
        return None  # or raise an error if you prefer

def get_temperatures(directory):
    temperatures = []
    temperatureList = []
    file_paths = glob.glob(os.path.join(directory, '*.csv'))
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        temperature_str = extract_temperature_from_filename(filename)
        temperature = float(temperature_str) if temperature_str is not None else None
        if temperature is None:
            print(f"Warning: No temperature found in filename {filename}. Skipping...")
            continue
        temperatures.append(temperature)
        temperatureList.append((temperature, file_path))
    # Sort files based on b-field
    temperatureList.sort(key=lambda x: x[0])
    temperatures.sort()
    return temperatureList, temperatures

def quick_plot(title : str, x_var : list, y_var : list, 
               x_label : str, y_label : str, 
               highdef : bool = False) -> None:
    if highdef == True:
        plt.figure(dpi=400)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_var, y_var)
    plt.show()

def plot_with_fit(title : str, 
                  x_var : list, 
                  y_var : list, 
                  fit : list, 
                  x_label : str = "Energy (eV)", 
                  y_label : str = '', 
                  highdef : bool = False):
    if highdef == True:
        plt.figure(dpi=400)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_var, y_var)
    plt.plot(x_var, fit)
    plt.show()

def select_file():
    """ 
    Opens a file dialog and returns path to the selected file. 
    Does not work in interactive windows.
    """
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.update()

    file_path = filedialog.askopenfilename(title="Select a file")

    root.destroy()

    if file_path:
        return file_path
    else:
        print("No file selected.")
        return None

def get_heatmap_data(energy, intensity, 
                     plot_with_mask : bool = False, 
                     apply_smooth : bool = False, 
                     min_energy=None, 
                     max_energy=None):
    from scipy.signal import savgol_filter as smooth
    if plot_with_mask:
        if min_energy is None or max_energy is None:
            raise ValueError("min_energy and max_energy must be specified when plot_with_mask is 1.")
        yvar, intensity = mask(energy, intensity, min_energy, max_energy)
    else:
        yvar = energy
    zvar = intensity
    if apply_smooth:
        zvar = smooth(zvar, 5, 3)
    return yvar, zvar

def plot_heatmap(xvariable, yvariable, zvariable, plotTitle, cmap='inferno', colorbar=False) -> None:
    """
    Function to plot a heatmap of z data as a function of x and y.  
    xvariable will go on the horizontal axis and should be a single list of values. 
    yvariable will go on the vertical axis and should be a list of identical lists containing the y values.
    The length of yvariable should be equal to the length of xvariable. 
    to the number of xvariable values. 
    zvariable should be a list of lists where each inner list corresponds to the intensity values for an xvariable value. 
    The length of zvariable should be the same length as yvariable."""
    # Convert lists to arrays
    xvariable_array = np.array(xvariable)
    yvariable_array = np.array(yvariable)
    zvariable_array = np.array(zvariable)

    # Create meshgrid of wavelength and magnetic field values
    x_grid, y_grid  = np.meshgrid(xvariable_array, yvariable_array[0])

    # Plot contour plot for each magnetic field
    plt.figure(dpi = 400, figsize=(4, 4))
    contour = plt.contourf(x_grid, y_grid, zvariable_array.T, cmap=cmap)
    if colorbar == True:
        plt.colorbar(contour, label='Intensity')

    plt.xlabel(r'$\mu_0H$ (T)', size = 16)
    plt.ylabel('Energy (eV)', size = 16)
    # plt.title(plotTitle, size = 16)
    plt.show()

def plot_data_array(data: xr.DataArray, title: str = '', save=False, save_path: str = '') -> None:
    """ Plots an xarray DataArray with time on x-axis and values on y-axis. 
    Written for transient absorption data.
    Assumes time is in ps and values are in arbitrary units. """
    x_vals = data.coords['time'].values
    y_vals = data.values
    plt.plot(x_vals, y_vals)
    plt.xlabel("Time (ps)")
    plt.ylabel("$R/R_0$")
    plt.title(title)
    if save:
        if save_path is None:
            raise ValueError("save_path must be specified when save is True.")
        plt.savefig(save_path)
    # plt.xscale('log')
    plt.show()

