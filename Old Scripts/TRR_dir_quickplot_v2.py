"""
Created March 24, 2025

@author: piper

This script takes a folder of PREPROCESSED transient absorption data and makes 
quick plots of the raw data with the file name as the plot title. 
For quickly looking over data.

Long file paths can cause problems, so it's helpful to have file names as short as possible.
This is because of some limitation that Windows has.
"""
# %%

import os
from os.path import dirname, join, basename
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr

# Find code directory relative to our directory to import correct version of pipersfunctions.
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

from pipersfunctions import plot_data_array

import TRR_functions as trr

if __name__ == "__main__":
    dir_relative_path = r"..\TRR\processed_data\2025-08-28"
    directory = join(THIS_DIR, dir_relative_path)

    file_paths_list = [join(directory, f) for f in os.listdir(directory) if '.csv' in f]

    save_directory = join(directory, 'quickplots')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    for filepath in file_paths_list:
        filename = basename(filepath)
        save_path = join(save_directory, filename.replace('.csv', ''))

        # Create a data array with time in ps
        data_array = trr.prepare_data_array(filepath, inputtype = 'CSV')
        if data_array.size == 0:
            print(f"Warning: No data found in file {filename}. Skipping...")
            continue
        if os.path.exists(save_path):
            print(f"Warning: File {filename} already exists in the save directory. Skipping...")
            continue
        # plot_data_array(dataArray, title = "%s" % filename, save = True, save_path = save_path)
        plt.plot(data_array.time.values, data_array.values)
        plt.title(f'{filename} processed')
        plt.xlabel('Time (ps)')
        plt.ylabel('$R/R_0$')
        plt.savefig(save_path)
        plt.show()

# %%
