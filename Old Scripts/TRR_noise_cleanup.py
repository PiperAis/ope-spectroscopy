""" 
Created March 23, 2025
Updated September 4, 2025

author: @piper 

Version 3.1

3.0: Combines v1 and v2 for more efficient processing.
3.1: Uses config file to designate directory paths

Does initial processing of TRR data: background subtraction,
normalization, and removing noise (single large blips), then saves
the processed data into the processed_data subdirectory with the same
filenames as the raw data.

This preserves the metadata (bfield, flake name, and other parameters) 
which can be saved as attrs of the DataArrays when read.

To reload saved csv files, use
data_array = trr.prepare_data_array(filepath, inputtype = 'CSV')

Doesn't handle subdirectories within the main data folder you are 
processing. If you have subdirectories, process the main directory 
first, then inner folders. This should work okay, but might cause 
weirdness with folder names. The best thing to do is not have sub-
folders.
"""

# %%

import os
from os.path import dirname, join, basename, isfile
import sys
import numpy as np
import matplotlib.pyplot as plt

# Find code directory relative to our directory to import correct 
# version of custom modules.
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import processing_package as pp
import config

data_directory = config.DATA_DIR
processed_data_directory = config.PROCESSED_DATA_DIR
plots_directory = config.PLOTS_DIR

def update_plot():
    """Update the plot with selected dip regions."""
    ax.clear()
    ax.scatter(data_array.time, data_array_original, 
               label="Original Data", s=6, alpha=0.5)
    ax.plot(data_array.time, data_array_original, 
            label="Original Data", alpha=0.5)
    ax.plot(data_array.time, data_array, 
            label="Data with Removed Dips", linestyle="dotted",
            alpha=0.7)
    ax.set_title("Click to select dip edges (right-click to undo)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.canvas.draw()

def onclick(event):
    """Handle mouse clicks: left-click to select, right-click to undo."""
    global data_array

    if event.xdata is not None:
        # Find the closest index to the clicked x-coordinate (time)
        idx = np.argmin(np.abs(data_array.time.values - event.xdata))
        selected_time = data_array.time[idx].values

        if event.button == 1:  # Left-click to add
            selected_points.append(idx)
            selected_times.append(selected_time)
            print(f"Selected index: {idx}, Time: {data_array.time[idx].values}, Value: {data_array[idx].values}")

            # If we have an even number of selections (pairs), fill in NaNs
            if len(selected_points) % 2 == 0:
                start, end = selected_points[-2], selected_points[-1]
                if start > end:
                    start, end = end, start  # Ensure correct order
                data_array[start:end+1] = np.nan
                print(f"Marked indices {start} to {end} for removal")

        elif event.button == 3 and len(selected_points) >= 2:  # Right-click to undo last pair
            removed_start = selected_points.pop()
            removed_end = selected_points.pop()
            if removed_start > removed_end:
                removed_start, removed_end = removed_end, removed_start
            print(f"Undo selection: indices {removed_start} to {removed_end}")

            # Restore original values in the removed region
            data_array[removed_start:removed_end+1] = data_array_original[removed_start:removed_end+1]

        update_plot()

if __name__ == "__main__":
    # Get the directory where raw data is stored
    expt_date = r"2025-08-26"
    data_directory = join(data_directory, expt_date)

    # Extract a list of file paths
    file_paths_list = [join(data_directory, f) for f in os.listdir(data_directory) 
                       if isfile(join(data_directory, f)) and not f.__contains__(".")]

    # Set a directory to save processed data in
    save_directory = data_directory.replace('raw_data', 'processed')  

    for filepath in file_paths_list:
        filename = basename(filepath)

        # Create a data array
        data_array = pp.prepare_data_array(filepath, inputtype='TimeSpec')
        data_array = data_array.sel(time=~data_array.get_index("time").duplicated())

        # Plot the raw data
        plt.plot(data_array.time.values, data_array.values)
        plt.title(f'{data_array.attrs['sample']} {data_array.attrs['field']}T raw data')
        plt.xlabel('Time (ps)')
        plt.ylabel('Signal (V)')
        plt.show()

        # Identify and remove noise blips
        selected_points = []
        selected_times = []

        # Keep an original copy for restoring values
        data_array_original = data_array.copy()

        # Create interactive plot
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.canvas.mpl_connect("button_press_event", onclick)
        update_plot()
        plt.show()
        selected_times = [data_array.time[ind].values for ind in selected_points]
        final_selected_times = [float(t_val) for t_val in selected_times]

        print("Final selected indices:", selected_points)
        print("Final selected times:", final_selected_times)

        # Plot results of noise removal
        plt.figure(figsize=(10, 5))
        plt.plot(data_array_original.time, data_array_original, 
                 label="Original Data", linestyle="dotted", alpha=0.5)
        plt.plot(data_array.time, data_array, 
                 label="Cleaned Data", linewidth=1)
        plt.legend()
        plt.show()

        # Ensure scale factor and rzero attrs exist (don't assume keys are present).
        # Use `scale_factor_value` (non-reserved) for persistence in NetCDF.
        if ('scale_factor' not in data_array.attrs) and ('scale_factor_value' not in data_array.attrs):
            print("Scale factor not included in file name, setting to 1.")
            data_array.attrs['scale_factor_value'] = 1

        if 'rzero' not in data_array.attrs and 'r_zero' not in data_array.attrs:
            print("R_0 not included in file name, setting to 1.")
            data_array.attrs['rzero'] = 1

        # Background subtract and normalize
        data_array = pp.subtract_background(data_array, threshold = -2)
        data_array = pp.divide_out_factors(data_array)

        # Create proper file structure for saving processed data
        save_path = join(processed_data_directory, f'{filename}.nc')
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # save data array
        data_array.to_netcdf(path = save_path)

#%% 


