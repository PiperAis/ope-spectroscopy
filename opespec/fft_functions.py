"""
Helper functions for fast fourier transforms of TRR data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import xarray as xr

def compute_fft(time_vals, data):
    """ Takes in time values and data, computes the FFT and returns the frequencies and magnitudes 
    Does not require a uniform time grid but uses the mean of the time steps to compute the FFT. """
    dt = np.mean(np.diff(time_vals))
    N = len(data)
    frequencies = np.fft.fftfreq(N, d=dt)
    freqs = frequencies[frequencies > 0]
    fft_result = np.fft.fft(data)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude = fft_magnitude[frequencies > 0]
    return freqs, fft_magnitude

def fourier_transform(data: xr.DataArray):
    """ Computes FFT of data given in a DataArray. Requires uniform time steps? """
    y_vals = data.values
    dt = np.mean(np.diff(data.coords['time'].values))  # Compute time step
    fft_result = np.fft.fft(y_vals)  # Compute FFT
    fft_magnitude = np.abs(fft_result)  # Magnitude of FFT
    freqs = np.fft.fftfreq(len(y_vals), d=dt)  # Compute frequency axis
    return freqs, fft_magnitude

def plot_fft(freqs, fft_magnitude):
    plt.figure(figsize=(8,5))
    plt.plot(freqs, fft_magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Fourier Transform (Frequency Spectrum)")
    plt.xlim([0, np.max(freqs)])  # Only show positive frequencies
    plt.grid()
    plt.show()

def apply_hanning_window(data):
    """ Input is a 1D array of data. Returns the data multiplied by a Hanning window. """
    window = np.hanning(len(data))
    return data * window

def uniform_time_grid(time_vals, y_vals, num_points=1000):
    """ Takes time values and returns a uniform time grid with the same or more points """
    if num_points < len(time_vals):
        num_points = len(time_vals)
    t_uniform = np.linspace(time_vals.min(), time_vals.max(), num_points)
    interp_func = interp1d(time_vals, y_vals, kind='linear', fill_value="extrapolate") # type: ignore
    y_uniform = interp_func(t_uniform)
    return t_uniform, y_uniform

def interpolate_uniform_grid(data_array, num_points):
    t_uniform, y_uniform = uniform_time_grid(time_vals = data_array.time.values, 
                                            y_vals = data_array.values, 
                                            num_points = num_points)
    
    data_array = xr.DataArray(data = y_uniform, coords = {'time': t_uniform})
    return data_array