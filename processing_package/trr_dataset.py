""" 
Helper class for TRR (time resolved reflectance) data
collected using TimeSpec software. Expects file naming conventions 
that use hyphens to separate pieces of information: \n

[Dataset number]-flake[sample name]-[bfield]T-[scale factor]SF-[R0]mV \n
Separate any further metadata with hyphens and include units after the 
value. Optionally, include an identifier before the value - planning to
implement handling for user defined metadata in filename in the future.

Use n to designate negative fields and p instead of . in the file name. \n
Example file name:\n
01-flake1a-n0p2T-180kSF-118mV-pumpwl532nm-pumppwr20uW

\n\n
V 1.0

author: @piper
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
import re
from scipy.optimize import curve_fit
import xarray as xr
from xarray import Dataset
from typing import Self

xr.set_options(keep_attrs=True)

from . import fitting_functions as ff
import processing_package as pp

class TRRDataset(Dataset):
    __slots__ = (
        'rawfilename',
        'set_number', 
        'sample',
        'field',
        'scale_factor',
        'rzero',
        'filepath'
    ) 

    def __init__(self, filepath : pathlib.PurePath, input_file_type : str = 'TimeSpec'):
        
        if filepath.suffix != '.nc':
            array = self.gen_from_file(filepath, input_file_type)
            super().__init__(data_vars={'raw': array}, attrs=array.attrs)
            self.rawfilename = filepath.name
        else:
            # Handle .nc files
            ds = xr.open_dataset(filepath)
            super().__init__(ds.data_vars, ds.coords, ds.attrs)
            self.attrs.update(ds.attrs)
            
        self.set_number = self.__getattr__('set number')
        self.field = self.__getattr__('field')
        self.sample = self.__getattr__('sample')
        self.scale_factor = self.__getattr__('scale factor')
        self.rzero = self.__getattr__('rzero')
        self.filepath = str(filepath)
        
        self.load()
        self.close()

        return None
    
    def gen_from_file(self, filepath : pathlib.PurePath, input_file_type: str = 'TimeSpec'):

        # Set parameters for import for pandas read_csv function
        if input_file_type == 'TimeSpec':
            usecols = [0, 1]
            separator = r'\s+'
            header = None
            time_index = 0
            data_index = 1
            time_factor = 10**(-3)
        if input_file_type == 'CSV':
            usecols = None
            separator = ','
            header = 'infer'
            time_index = 'time'
            time_factor = 1
        filepath_string = str(self.filepath)

        raw_data = pd.read_csv(filepath_string, 
                            usecols = usecols, 
                            sep = separator, 
                            header = header)
        if input_file_type == 'CSV':
            data_index = raw_data.columns.tolist()[1]
        
        # Get metadata from filename
        attrs = {'field' : self.get_field() if self.get_field() is not None else 0,
                      'sample' : self.get_sample_name(),
                      'scale factor' : self.get_scale_factor(),
                      'rzero' : self.get_rzero(),
                      'datafile' : filepath.name,
                      'set number' : filepath.name.split('-')[0]
                      }

        array = xr.DataArray(data = raw_data[data_index], 
                        coords = {'time' : raw_data[time_index]*time_factor},
                        attrs = attrs,
                        name = 'raw')
        
        return array

    def get_field(self) -> float | None:
        """ 
        Gets magnetic field from data filename, returning it as a str or 
        None if not found. Assumes filename structure is (e.g.)
        ***-n2p6T-*** where n represents negative sign, p represents a
        period, and the field is separated from other information by '-'
        """
        # Search for the regex pattern denoting bfield
        match = re.search(r'-n?\d+(p\d+)?T', str(self.rawfilename))

        # Convert filename formatting to float
        if match:
            bstring = match.group(0)
            bstring = bstring.replace('-', '')
            bstring = bstring.replace('p', '.')
            bstring = bstring.replace('n', '-')
            bstring = bstring.replace('T', '')
            bfield = float(bstring)
            return bfield
        else:
            return None

    def get_sample_name(self) -> str:
        """ Gets sample name from data filename, returning 'unnamed
        sample' if not found. 
        """
        # Search for the regex pattern denoting flake name
        match = re.search(r'flake.*-', str(self.rawfilename))

        # Format sample name
        if match:
            sample_name = match.group(0)
            sample_name = sample_name.split('-')[0]
            sample_name = sample_name.replace('-', '')
            sample_name = sample_name.replace('flake', 'Flake ')
            return sample_name
        else:
            return 'unnamed sample'

    def get_scale_factor(self) -> int:
        """ 
        Gets scale factor from data filename and returns it as an int. 
        Returns 1 if pattern is not found. 
        """
        # Search for regex pattern denoting scale factor
        match = re.search(r'-\d+kSF', str(self.rawfilename))

        # Format scale factor
        if match:
            scale_factor = match.group(0)
            scale_factor = scale_factor.replace('-', '')
            scale_factor = scale_factor.replace('SF', '')
            scale_factor = scale_factor.replace('k', '000')
            scale_factor = int(scale_factor)
            return scale_factor
        else:
            return 1

    def get_rzero(self) -> float:
        """ 
        Gets r_0 from data filename and returns it as a float. Returns 1 if 
        pattern is not found. 
        """
        # Search for regex pattern denoting r_0
        match = re.search(r'-\d+p?(\d+)?mV', str(self.rawfilename))

        # Format value for return
        if match:
            rzero = match.group(0)
            rzero = rzero.replace('-', '')
            rzero = rzero.replace('mV', '')
            rzero = rzero.replace('p', '.')
            rzero = float(rzero)
            return rzero
        else:
            return 1

    def add_array(self, step_name : str, array : xr.DataArray):
        self[step_name] = array
        return

    def save(self, save_dir, filename_modifier : str = ''):
        if filename_modifier:
            save_path = save_dir / f'{self.set_number}{filename_modifier}.nc'
        else:
            save_path = save_dir / f'{self.set_number}.nc'
        self.to_netcdf(path = save_path, mode = 'a')
        return
    
    def remove_noise(self, processor):
        data_array = self[processor.previous_step].copy()
        data_array_original = self[processor.previous_step].copy()

        # Create interactive plot
        fig, ax = plt.subplots(figsize=(10, 5))
        remover = pp.NoiseRemover(data_array, data_array_original, ax, fig)
        fig.canvas.mpl_connect("button_press_event", remover.onclick)
        remover.update_plot()
        pp.add_noise_removal_buttons(fig, ax, remover)
        plt.show()

        final_selected_points = remover.selected_points

        plt.figure(figsize=(10, 5))
        plt.plot(data_array_original.time, data_array_original, 
                    label="Original Data", linestyle="dotted", alpha=0.5)
        plt.plot(data_array.time, data_array, 
                    label="Cleaned Data", linewidth=1)
        plt.legend()
        plt.xlabel("Time (ps)")
        plt.ylabel("Reflectance (arb. units)")
        plt.title(f"{self.set_number} noise removal")
        plt.savefig(fname = processor.get_plot_save_path(self))
        plt.clf()

        self.add_array(processor.current_step, data_array)

        return


def mask_data_array(
        data_array : xr.DataArray, 
        min_time : int | float, 
        max_time : int | float
        ) -> xr.DataArray:
    """ 
    Returns a DataArray with datapoints outside of the given time
    limits replaced with NaN. 
    """
    condition = (data_array >= min_time) and (data_array <= max_time)
    new_array = data_array.where(condition).all()
    
    return xr.DataArray(new_array)

# --- Functions for processing TRR data --- #

def subtract_background(
        data: xr.DataArray, 
        threshold: float
        ) -> xr.DataArray:
    """ 
    Finds average of data before t0 and subtracts from the entire 
    data set. Threshold sets the time before which the average is 
    calculated.
    """
    background_data = data.where(data.time < threshold, drop = True)
    average = (background_data.mean().item() if background_data.count() > 0 
               else None)
    if average is not None:
        data.values -= average
    return data

def divide_out_factors(data_array : xr.DataArray) -> xr.DataArray:
    """
    Normalize transient reflectance data by dividing out R0 and scale 
    factor. Expects input array to have attributes as assigned by the 
    prepare_data_array function.
    """
    # Support either 'rzero' or 'r_zero' keys and fall back to 1 if missing
    r_zero = data_array.attrs.get('rzero', data_array.attrs.get('r_zero', 1))
    # Prefer the non-reserved key `scale_factor_value`, fall back to
    # legacy `scale_factor` for compatibility.
    scale_factor = data_array.attrs.get(
        'scale_factor_value', data_array.attrs.get('scale_factor', 1))

    data_array.values = data_array.values / (r_zero * scale_factor)
    return data_array

def subtract_biexponential_decay(
        x_data, 
        y_data, 
        time_constant1 : int = 50, 
        time_constant2 : int = 175):
    """
    Fits a bi-exponential decay to the data and subtracts it, then 
    plots the original data with the fit and returns the corrected data.
    """

    initial_guess = (y_data[0], time_constant1, y_data[0]/2, time_constant2, 
                     np.mean(y_data))
    params, _ = curve_fit(ff.bi_exp_decay, x_data, y_data, p0=initial_guess)

    # Generate the fitted curve for all time points
    fitted_curve = ff.bi_exp_decay(x_data, *params)

    # Subtract fitted curve from original data
    y_corrected = y_data - fitted_curve

    plt.plot(x_data, y_data)
    plt.plot(x_data, fitted_curve)
    plt.title("Fit")
    plt.show()

    # Return new data array
    return y_corrected