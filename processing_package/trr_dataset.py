""" 
Helper class (xr accessor) for TRR (time resolved reflectance) data
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
V 1.5


To Do
- Add units

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

xr.set_options(keep_attrs=True)

from . import fitting_functions as ff
import processing_package as pp

@xr.register_dataset_accessor("trrxr")
class TRRDataset:
    """
    An extension for an XArray dataset for TRR data processing workflow.
    """

    def __init__(self, xarray_obj):
        
        self._obj = xarray_obj
        self._verify_attributes()
        
        self._obj.load()
        self._obj.close()

        return None
    
    @classmethod
    def create_trr_dataset(cls, filepath : pathlib.PurePath | None = None, input_file_type : str = 'TimeSpec'):

        def gen_from_file(filepath : pathlib.PurePath, input_file_type: str = 'TimeSpec'):

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

            filepath_string = str(filepath)
            raw_data = pd.read_csv(filepath_string,
                                   usecols = usecols,
                                   sep = separator,
                                   header = header)

            if input_file_type == 'CSV':
                data_index = raw_data.columns.tolist()[1]

            array = xr.DataArray(data = raw_data[data_index],
                            coords = {'time' : raw_data[time_index]*time_factor},
                            attrs = {'data filename':filepath.stem},
                            name = 'raw')

            return array

        if filepath == None:
            raise TypeError("Pass a filepath to generate dataset.")

        if filepath.suffix == '.nc':
            raise ValueError("This function is for generating a dataset from a "
            ".csv or TimeSpec file. Use XArray builtins to load an existing .nc file.")

        array = gen_from_file(filepath, input_file_type)
        dataset = xr.Dataset(data_vars = {'raw' : array},
                            attrs = array.attrs)

        return dataset
    
    @property
    def set_number(self):
        return self._obj.attrs.get('set number')

    def _init_set_number(self):
        set_string = self._obj.attrs['data filename'].split('-')[0]
        self._obj.attrs['set number'] = set_string

    @property
    def bfield(self):
        return self._obj.attrs.get('bfield')

    def _init_bfield(self):
        """
        Gets magnetic field from data filename. Assumes filename structure is
        (e.g.) ***-n2p6T-*** where n represents negative sign, p represents a
        period, and the field is separated from other information by '-'.
        """
        match = re.search(r'-n?\d+(p\d+)?T', self._obj.attrs['data filename'])
        if match:
            bstring = match.group(0)
            bstring = bstring.replace('-', '')
            bstring = bstring.replace('p', '.')
            bstring = bstring.replace('n', '-')
            bstring = bstring.replace('T', '')
            self._obj.attrs['bfield'] = float(bstring)
        else:
            self._obj.attrs['bfield'] = float(0)

    @property
    def sample(self):
        return self._obj.attrs.get('sample')

    def _init_sample(self):
        """ Gets sample name from data filename, returning 'unnamed sample' if not found. """
        match = re.search(r'flake.*-', self._obj.attrs['data filename'])
        if match:
            sample_name = match.group(0)
            sample_name = sample_name.split('-')[0]
            sample_name = sample_name.replace('-', '')
            sample_name = sample_name.replace('flake', 'Flake ')
            self._obj.attrs['sample'] = sample_name
        else:
            self._obj.attrs['sample'] = 'unnamed sample'

    @property
    def sf(self):
        return self._obj.attrs.get('sf')

    def _init_sf(self):
        """ Gets scale factor from data filename as an int. Returns 1 if pattern not found. """
        match = re.search(r'-\d+kSF', self._obj.attrs['data filename'])
        if match:
            scale_factor = match.group(0)
            scale_factor = scale_factor.replace('-', '')
            scale_factor = scale_factor.replace('SF', '')
            scale_factor = scale_factor.replace('k', '000')
            self._obj.attrs['sf'] = int(scale_factor)
        else:
            self._obj.attrs['sf'] = 1

    @property
    def rzero(self):
        return self._obj.attrs.get('rzero')

    def _init_rzero(self):
        """ Gets r_0 from data filename as a float. Returns 1 if pattern not found. """
        match = re.search(r'-\d+p?(\d+)?mV', self._obj.attrs['data filename'])
        if match:
            rzero = match.group(0)
            rzero = rzero.replace('-', '')
            rzero = rzero.replace('mV', '')
            rzero = rzero.replace('p', '.')
            self._obj.attrs['rzero'] = float(rzero)
        else:
            self._obj.attrs['rzero'] = 1

    def _verify_attributes(self):
        
        required_attributes = ['data filename', 'set number', 'sample', 'bfield', 'rzero', 'sf']
        for item in required_attributes:
            if item in self._obj.attrs.keys():
                required_attributes.remove(item)

        if 'data filename' in required_attributes:
            raise NameError("Something went wrong when initially loading your dataset, the data filename did not get saved as an attr.")
        
        if len(required_attributes) > 0:
            self._init_set_number()
            self._init_bfield()
            self._init_sample()
            self._init_rzero()
            self._init_sf()

        return

    def add_array(self, step_name : str, array : xr.DataArray):
        self._obj[step_name] = array
        return

    def save(self, save_dir, filename_modifier : str = ''):
        if filename_modifier:
            save_path = save_dir / f'{self.set_number}{filename_modifier}.nc'
        else:
            save_path = save_dir / f'{self.set_number}.nc'
        self._obj.to_netcdf(path = save_path, mode = 'w')
        return
    
    def remove_noise(self, processor):
        data_array = self._obj[processor.previous_step].copy()
        data_array_original = self._obj[processor.previous_step].copy()

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
        plt.savefig(fname = processor.get_plot_save_path(self._obj))
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
        'sf', data_array.attrs.get('scale_factor_value', data_array.attrs.get('scale_factor', 1)))

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