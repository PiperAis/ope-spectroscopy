"""
Updated March 18, 2025

Make sure the correct version of pipersfunctions.py is in the directory
with this script. If you found this script in a Code directory inside a 
project folder on Box, the correct version of pipersfunctions is in the 
same directory. 

Otherwise, you can check version numbers but I'm new at versioning so that 
might not be reliable.

This script should be stored in the Code directory of the project folder 
it processes data for. 

@author: piper
"""

#%% 
"""Import Packages"""

# Find code directory relative to our directory
# to import correct version of pipersfunctions.
from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import pipersfunctions as pf

# Other packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize
from scipy.signal import savgol_filter as smooth

""" 
--Start of user input--

Replace with relative path to directory where CSV files are stored.
File names within this directory should include b fields in the format
"XXT", where XX is the b field in Tesla. You can include other information
in the filename as long as the b field is separated from other items 
by an underscore or dash. It will convert 'p' to '.' and 'n' to '-'
"""
# dir_relative_path = r"..\PL\raw_data\2025-04-29\flakeD1-bsweep"
# dir_relative_path = r"..\PL\raw_data\2025-04-30\D2-bsweep"
dir_relative_path = r"..\PL\raw_data\2025-04-30\D8-bsweep"

''' Plot settings '''
plot_title = ''
titleFontSize = 15
axisFontSize = 12

""" Set to 1 to apply energy bounds """
plot_with_mask = 1

min_energy = 1.3
max_energy = 1.35

""" Set to 1 to plot each spectrum individually as well """
plotTraces = 0

""" Applies a Savitzky-Golay filter to the data to smooth it."""
applySmooth = 0

"""Normalize the data to the maximum value from the entire dataset."""
normalize_all = 0

""" --End of user input-- """

# Get list of all CSV files in the directory
directory = join(THIS_DIR, dir_relative_path)
file_paths = glob.glob(os.path.join(directory, '*'))

# Initialize lists to store data
intensitydata = []
xvar_all=[]
yvar_all=[]
maxes = []

# Generate list of b-fields from file names
bfields, bfield_list = pf.get_b_fields(file_paths)

# Read data from each file
for i, (bfield, file_path) in enumerate(bfield_list):
    # Read wavelength and intensity data from file
    energy, intensity, max_intensity = pf.read_spectrum(file_path, 
                                                        umBlaze = 0, 
                                                        normalize = 1, 
                                                        xEnergy = 1)
    maxes.append(max_intensity)
    # convert into data format for heatmap and apply settings
    yvar, zvar = pf.get_heatmap_data(energy, 
                                     intensity, 
                                     plot_with_mask = plot_with_mask, 
                                     min_energy = min_energy, 
                                     max_energy = max_energy, 
                                     apply_smooth = applySmooth)
    yvar_all.append(yvar)  
    intensitydata.append(zvar)


    # Plot each trace individually
    if plotTraces == 1:
        plt.figure(dpi=400)
        plt.title("%s T" % bfields[i])
        plt.plot(yvar, zvar)
        plt.show()   

if normalize_all == 1:
    intensitydata = intensitydata/max(maxes)

# Create contour plot
pf.plot_heatmap(bfields, yvar_all, intensitydata, plot_title, cmap='inferno')

# %%
