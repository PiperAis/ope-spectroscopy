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
import os
from os.path import dirname, join, basename, isfile
from os import listdir
import pandas as pd
import pathlib
from pathlib import Path
import sys
import xarray as xr

# Add code directory to system path to import custom modules.

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))

# Import custom modules

import processing_package as pp
import config

def run_noise_removal_process(filepath : Path):
    filename = filepath.name
    set_number = filename.split('-')[0]
    plot_save_path = plots_dir / f"{set_number}-noiseremoved.png"

    # Create a data array
    data_array = pp.prepare_data_array(str(filepath), inputtype='TimeSpec')

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
    plt.savefig(fname = str(plot_save_path))
    plt.show()

    # Save data array in format usable by xarray module for next 
    # processing stage
    save_path = save_dir / f'{set_number}.nc'
    if not save_dir.exists():
        save_dir.mkdir()
    
    dataset = xr.Dataset({'raw' : data_array_original, 'noise removed': data_array})

    dataset.attrs = data_array_original.attrs
    
    try:
        dataset.to_netcdf(path = str(save_path))
    except Exception:
        print("Error saving DataArray to NetCDF. Check if path length is too long.")
        print(f"save_path was: {str(save_path)}")

    '''Kind of a hacky way of finding the plot path relative to the
    report path. Perhaps can improve later. File extension needs
    to be included for rendering .md report file properly.'''
    plot_rel_path = f"..{str(plot_save_path).split(r"TRR")[-1]}"

    pp.update_report(report_file, filename, section = "noise corrected", plot_path = f"{plot_rel_path}") # type: ignore
    
    return None

# Create directory paths from config file and experiment date

raw_data_dir = config.DATA_DIR / expt_date
processed_data_dir = config.PROCESSED_DATA_DIR / expt_date
if not processed_data_dir.exists():
    processed_data_dir.mkdir()
reports_dir = config.REPORTS_DIR
if not reports_dir.exists():
    reports_dir.mkdir()
plots_dir = config.PLOTS_DIR / expt_date
if not plots_dir.exists():
    plots_dir.mkdir(parents = True)

# initialize a state variable to track processing progress
# and define directory paths accordingly.
# maybe eventually modify initialize next step func. to
# do this initial generation too?
state = {'step_number' : 0, 
        'step_name' : 'pre-init',
        'steps_list' : ['noise corrected', 'processed', 'fourier transform'],
        'raw_data_dir' : raw_data_dir,
        'load_dir' : raw_data_dir, 
        'save_dir': processed_data_dir,
        'plots_dir' : plots_dir,
        'reports_dir' : reports_dir
        }

""
if __name__ == "__main__":

    # Generate outline for markdown report
    report_file = pp.generate_skeleton(state)

    # --- Step 1: Remove noise --- #

    # Set directories
    # state = pp.initialize_next_step(state, steps_list)
    load_dir = state['load_dir']
    save_dir = state['save_dir']

    # Get list of files in the given experiment folder
    file_paths_list = [f for f in load_dir.iterdir() if not f.name.__contains__(".")]

    for filepath in file_paths_list:
        run_noise_removal_process(filepath)
        
    # Step 2: Processing (normalize and subtract decay)
    
    state = pp.initialize_next_step(state)

    load_dir = state['load_dir']
    save_dir = state['save_dir']

    if not save_dir.exists():
        save_dir.mkdir()

    # Create list of file paths in load directory
    ## Need to update everything to use pathlib instead of os.path. So far just done thru this point and report generator function.
    file_paths_list = [str(f) for f in load_dir.iterdir() if f.is_file()]

    for filepath in file_paths_list:
        pp.normalize_data(filepath, save_dir)
        subtraction_results = pp.subtract_decay(filepath, str(plots_dir), save_dir, str(report_file))

# %%
