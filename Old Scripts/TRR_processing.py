"""
Created March 24, 2025
Updated September 4, 2025

author: @piper

Version 1.1

This script works on csv files that have been preprocessed using the
TRR_noise_cleanup script.

Subtracts exponential decay, or constant value if fit fails.

In progress
"""
# %%

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import os
from os.path import dirname, join, basename
from os import listdir
import pandas as pd
import sys
import xarray as xr

# Find code directory relative to our directory to import correct version of pipersfunctions.
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import processing_package as pp
import config

data_directory = config.DATA_DIR
processed_data_directory = config.PROCESSED_DATA_DIR
plots_directory = config.PLOTS_DIR

if __name__ == "__main__":

    # Prepare data array from input file and save sample and field parameters
    expt_date = r"2025-08-26"
    data_directory = join(processed_data_directory, expt_date)
    
    file_paths_list = [join(data_directory, f) for f in listdir(data_directory) 
                    if '.nc' in f]
    
    for filepath in file_paths_list:
        filename = basename(filepath)

        ds = xr.open_dataset(filepath)
        data_array = ds[list(ds.data_vars)[0]]
        
        print("hasattr values upon creation:", hasattr(data_array, "values"))

        data_array = data_array.sel(time=~data_array.get_index("time").duplicated())


        plt.figure(figsize = (12,4))
        plt.scatter(data_array.time, data_array.values, s=6)
        plt.title(f'{data_array.attrs['sample']} {data_array.attrs['field']} raw')
        plt.show()

        # Mask the data 
        min_time = 5
        max_time = data_array.time.values.max()

        # print("hasattr values after sel:", hasattr(data_array, "values"))

        data_array = pp.mask_data_array(data_array, min_time, max_time)
        # print("hasattr values after masking:", hasattr(data_array, "values"))
        # print('type of masked data array: ', type(data_array) )

        # Fit and subtract exponential decay
        # Fit and subtract exponential decay
        try:
            fit_results = pp.subtract_exponential_decay(data_array.time.values, data_array.values, time_constant = 0.01)
            subtracted_data = fit_results[0]
            subtracted_data -= np.nanmean(subtracted_data)
            time_constant = fit_results[1]
        except:
            print('Exponential fit failed, subtracting DC component.')
            subtracted_data = data_array.values - np.nanmean(data_array.values)

        residual_array = xr.DataArray(
        subtracted_data,
        dims=data_array.dims,
        coords=data_array.coords,
        attrs=data_array.attrs,
        name=data_array.name
        )

        # Make time grid uniform 
        # residual_array = interpolate_uniform_grid(data_array, num_points = 1000)
        residual_array.name = 'res'
        plt.figure(figsize = (12,2))
        plt.scatter(residual_array.time, residual_array.values, s=6)
        plt.title(f'{data_array.attrs['sample']} {data_array.attrs['field']} processed')
        plt.show()

        plt.figure(figsize = (12,4))
        plt.plot(residual_array.time, residual_array.values)
        plt.title(f'{data_array.attrs['sample']} {data_array.attrs['field']} processed')
        plt.show()

        save_directory = join(data_directory, r'step2')
        save_path = join(save_directory, f'{filename}')

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        data_set = xr.Dataset({'data' : data_array, 'residuals' : residual_array}, attrs = data_array.attrs)
        data_set.to_netcdf(path = save_path)
#%%