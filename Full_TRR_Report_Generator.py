'''
Runs all of the TRR processing scripts and generates a report.

Noise removal step does not work if run in an interactive window
in the environment.

Created 9/04/2025 by Piper Aislinn

Version 1.0
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.optimize import curve_fit
import os
from os.path import dirname, join, basename, isfile
from os import listdir
import pandas as pd
import sys
import xarray as xr

# Add code directory to system path to import custom modules.

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

# Import custom modules

import processing_package as pp
import config

def run_noise_removal_process(filepath):
    filename = basename(filepath)
    set_number = filename.split('-')[0]
    plot_save_path = f"{join(plots_dir, set_number)}-noiseremoved.png"

    # Create a data array
    data_array = pp.prepare_data_array(filepath, inputtype='TimeSpec')

    # Save raw data file name as an attribute of the data array
    data_array.attrs['raw data filename'] = filename
    
    # Keep an original copy for restoring values
    data_array_original = data_array.copy()

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(10, 5))
    remover = pp.NoiseRemover(data_array, data_array_original, ax, fig)
    fig.canvas.mpl_connect("button_press_event", remover.onclick)
    remover.update_plot()
    pp.add_noise_removal_buttons(fig, ax, remover)
    
    plt.show()

    # Save selected points/times in case needed
    final_selected_points = remover.selected_points
    final_selected_times = [float(data_array.time[ind].values) for ind in final_selected_points]

    # Plot results of noise removal
    plt.figure(figsize=(10, 5))
    plt.plot(data_array_original.time, data_array_original, 
                label="Original Data", linestyle="dotted", alpha=0.5)
    plt.plot(data_array.time, data_array, 
                label="Cleaned Data", linewidth=1)
    plt.legend()
    plt.xlabel("Time (ps)")
    plt.ylabel("Reflectance (arb. units)")
    plt.title(f"{set_number} noise removal")
    plt.savefig(fname = f"{plot_save_path}")
    plt.show()

    # Save data array in format usable by xarray module for next 
    # processing stage
    save_path = join(save_dir, f'{set_number}.nc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    dataset = xr.Dataset({'raw' : data_array_original, 'noise removed': data_array})

    dataset.attrs = data_array_original.attrs
    
    try:
        dataset.to_netcdf(path = save_path)
    except Exception:
        import traceback
        print("Error saving DataArray to NetCDF:")
        traceback.print_exc()
        print(f"save_path was: {save_path}")

    '''Kind of a hacky way of finding the plot path relative to the
    report path. Perhaps can improve later. File extension needs
    to be included for rendering .md report file properly.'''
    plot_rel_path = f"..{plot_save_path.split(r"TRR")[-1]}"

    pp.update_report(report_file, filename, section = "noise corrected", plot_path = f"{plot_rel_path}")
    
    return None

def subtract_decay(filepath):
    """ Takes a filepath (expects .nc filetype).
    Returns a list:
    [subtracted data (dataArray), residuals (dataArray)]

    Saves the processed data to existing dataset and adds the decay
    constant of the fit as an attribute of the dataset.
    """
    dataset = xr.open_dataset(filepath)

    # Load dataset into memory so .values is available, then close file to allow overwrite
    dataset.load()
    dataset.close()

    try:
        data_array = dataset['normalized'].copy()
    except:
        print("No variable named 'normalized' in the dataset. Check that your processed" \
        "data saved correctly during the normalization step. Canceling subtraction step.")
        return None

    # print(f"Data array attributes are {data_array.attrs}")

    data_array = data_array.sel(time=~data_array.get_index("time").duplicated())

    # Keep a copy of the original values to restore masked regions later
    original_values = data_array.values.copy()

    # Mask the data 
    min_time = data_array.time.values[data_array.argmax()]
    # min_time = 5
    max_time = data_array.time.values.max()
    max_time = 75
    # masked_da = pp.mask_data_array(data_array, min_time, max_time)
    masked_da = data_array.copy()
    masked_da = masked_da.where(data_array.time >= min_time)
    masked_da = masked_da.where(data_array.time <= max_time)
    masked_values = masked_da.values.copy()

    # Identify masked positions/indices
    mask = np.isnan(masked_values)

    # Get an array of the masked values and times with the NaNs dropped
    masked_values = masked_values[~mask]
    masked_times = masked_da.time.values[~mask]

    # Fit exponential decay to masked data, or subtract data avg if fit fails.
    initial_guess = [masked_values[0], 0.01, np.mean(masked_values)]

    try:
        params, _ = curve_fit(pp.exp_decay, 
                            masked_da.time, 
                            masked_da.values, 
                            p0=initial_guess, 
                            nan_policy='omit')
        
        fit_da = xr.DataArray(
            pp.exp_decay(data_array.time.values, *params),
            dims = data_array.dims,
            coords = data_array.coords,
            attrs = data_array.attrs, 
            name = 'fit')
        
        processed_masked = xr.DataArray(
            original_values - fit_da.values,
            dims = data_array.dims,
            coords = data_array.coords,
            attrs = data_array.attrs, 
            name = 'subtracted wonk'
        )

        fitted_data_array = processed_masked.where(data_array.time.values >= min_time)
        subtracted_data = fitted_data_array.values.copy()

        time_constant = params[0]

    except RuntimeError:
        print("Curve fit was unsuccessful. Subtracting DC component instead.")
        subtracted_data = masked_values - np.mean(masked_values)
        time_constant = 0
    
    # residual_values = masked_fit - masked_values

    # Create data array with subtracted data
    processed_data = xr.DataArray(
        subtracted_data,
        dims = data_array.dims,
        coords = data_array.coords,
        attrs = data_array.attrs,
        name = 'processed'
    )

    processed_data.plot() #type: ignore
    
    # Add time constant of subtracted decay to attributes
    processed_data.attrs['time_constant'] = time_constant

    # # Create a residual array
    # residual_array = xr.DataArray(
    #     residuals,
    #     dims=data_array.dims,
    #     coords=data_array.coords,
    #     attrs=data_array.attrs,
    #     name='residuals'
    # )

    # Add subtracted data and residuals to dataset
    dataset['subtracted'] = processed_data
    # dataset['residuals'] = residual_array
    dataset.attrs['time_constant'] = time_constant

    # Generate save path
    filename = basename(filepath)
    save_path = join(save_dir, filename)

    # Save dataset
    dataset.to_netcdf(path = save_path, mode = 'a')

    return [data_array]


# Create directory paths from config file and experiment date

raw_data_dir = join(config.DATA_DIR, expt_date)
processed_data_dir = join(config.PROCESSED_DATA_DIR, expt_date)
plots_dir = join(config.PLOTS_DIR, expt_date)
reports_dir = config.REPORTS_DIR

# list of processing steps for report generation
steps_list = ['noise corrected', 'processed', 'fourier transform']

# initialize a state variable to track processing progress
# and define directory paths accordingly.
# maybe eventually modify initialize next step func. to
# do this initial generation too?
state = {'step_number' : 0, 
        'load_dir' : raw_data_dir, 
        'save_dir': processed_data_dir,
        'plots_dir' : plots_dir,
        }

""
if __name__ == "__main__":

    # Generate outline for markdown report
    report_file = pp.generate_skeleton(raw_data_dir, steps = steps_list)

    # --- Step 1: Remove noise --- #

    # Set directories
    # state = pp.initialize_next_step(state, steps_list)
    load_dir = state['load_dir']
    save_dir = state['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get list of files in the given experiment folder
    file_paths_list = [join(load_dir, f) for f in os.listdir(load_dir) 
                       if isfile(join(load_dir, f)) and not f.__contains__(".")]

    for filepath in file_paths_list:
        run_noise_removal_process(filepath)
        
    # Step 2: Processing (normalize and subtract decay)
    
    state = pp.initialize_next_step(state, steps_list)

    load_dir = state['load_dir']
    save_dir = state['save_dir']

    # Create list of file paths in load directory
    file_paths_list = [join(load_dir, f) for f in os.listdir(load_dir)]

    for filepath in file_paths_list:
        pp.normalize_data(filepath, save_dir)
        subtraction_results = subtract_decay(filepath)

# %%
