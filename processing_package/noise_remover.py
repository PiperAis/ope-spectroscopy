"""
Defines a class to help with noise removal in TRR data, particularly
removing single- or two-point drops in the data due to chopper issues.

Updated 2025-09-09
author: @piper
"""

# import modules and functions
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.widgets import Button
from os.path import dirname, join, basename, isfile
from scipy.signal import savgol_filter as smooth

# import custom module config and set directory paths
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

from . import config

data_dir = config.DATA_DIR
processed_data_dir = config.PROCESSED_DATA_DIR
plots_dir = config.PLOTS_DIR
reports_dir = config.REPORTS_DIR

# --- Class used for noise cleanup --- #

class NoiseRemover:
    def __init__(self, data_array : xr.DataArray, data_array_original : xr.DataArray, ax, fig):
        self.data_array = data_array
        self.data_array_original = data_array_original
        self.selected_points = []
        self.selected_times = []
        self.ax = ax
        self.fig = fig

    def update_plot(self):
        self.ax.clear()
        self.ax.scatter(self.data_array.time, self.data_array_original, label="Original Data", s=6, alpha=0.5)
        self.ax.scatter(self.data_array.time, self.data_array, label="Data with Removed Noise", s=6, alpha=0.5)
        self.ax.plot(self.data_array.time, self.data_array, label="Data with Removed Noise", alpha=0.5)
        self.ax.set_title("Click to select dip edges (right-click to undo)")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.legend()
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.xdata is not None:
            idx = np.argmin(np.abs(self.data_array.time.values - event.xdata))
            selected_time = self.data_array.time[idx].values

            if event.button == 1:  # Left-click to add
                self.selected_points.append(idx)
                self.selected_times.append(selected_time)
                print(f"Selected index: {idx}, Time: {selected_time}, Value: {self.data_array[idx].values}")

                if len(self.selected_points) % 2 == 0:
                    start, end = self.selected_points[-2], self.selected_points[-1]
                    if start > end:
                        start, end = end, start
                    self.data_array[start:end+1] = np.nan
                    print(f"Marked indices {start} to {end} for removal")

            elif event.button == 3 and len(self.selected_points) >= 2:  # Right-click to undo last pair
                removed_start = self.selected_points.pop()
                removed_end = self.selected_points.pop()
                if removed_start > removed_end:
                    removed_start, removed_end = removed_end, removed_start
                print(f"Undo selection: indices {removed_start} to {removed_end}")
                self.data_array[removed_start:removed_end+1] = self.data_array_original[removed_start:removed_end+1]

            self.update_plot()
    
    def restart(self, event=None):
        """Reset noise removal for this file."""
        self.data_array[:] = self.data_array_original
        self.selected_points.clear()
        self.selected_times.clear()
        self.update_plot()
        print("Restarted noise removal for this file.")
    
    def accept(self, event=None):
        """Accept the current selection and close the plot."""
        plt.close(self.fig)


def add_noise_removal_buttons(fig, ax, remover):
    """
    Adds Restart and Accept buttons to the given figure for noise removal.
    """
    restart_ax = fig.add_axes([0.68, 0.01, 0.1, 0.05])
    restart_button = Button(restart_ax, 'Restart')
    restart_button.on_clicked(remover.restart)

    accept_ax = fig.add_axes([0.8, 0.01, 0.1, 0.05])
    accept_button = Button(accept_ax, 'Accept')
    accept_button.on_clicked(remover.accept)

    return restart_button, accept_button
