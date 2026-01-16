"""
Updated March 18, 2025

Make sure the correct version of pipersfunctions.py is in the directory with this script. If you found this script on Box,
the correct version is the version in the Code directory of the Box folder.

This script should be stored in the Code directory of the project folder it processes data for. 
@author: piper
"""

#%% 
"""Import Packages"""
# Find code directory relative to our directory to import correct version of pipersfunctions.
from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..', 'Code'))
sys.path.append(CODE_DIR)

# Other packages
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import minimize
import pipersfunctions as pf

#%% User Settings
# Add relative path where CSV files are stored
dir_relative_path = r"..\PL\raw_data\2025-04-30\D8-bsweep"
directory = join(THIS_DIR, dir_relative_path)

# Get list of all CSV files in the directory
file_paths = glob.glob(os.path.join(directory, '*.csv'))

# Plot settings
plotTitle = 'PL vs. B at 1.6 K'
titleFontSize = 15
yAxisLabel = "PL Intensity (arb units)"
axisFontSize = 12

#Set to 1 to set x-axis bounds
plot_with_mask = 1

min_energy = 1.26
max_energy = 1.4

# Fitting
doFit = 1
plotTraces = 1
applySmooth = 0

cmap='inferno'

#Fit parameters [amp1, cen1, wid1,...,ampN, cenN, widN]
p0 = [20/260, 1.31, 0.006, \
      90/260, 1.33, 0.006, \
      260/260, 1.34, 0.004, \
      #90/260, 1.345, 0.003, \
      80/260, 1.35, 0.005, \
      20/260, 1.355, 0.005]

bounds=[[0,None],[1.307,1.315],[0.001,0.1], \
        [0,None],[1.3,1.33],[0.001,0.01], \
        [0,None],[1.32,1.35],[0.001,0.01], \
        #[0,None],[1.327, 1.35],[0.001,0.01], \
        [0,None],[1.33,1.356],[0.001,0.01], \
        [0,None],[1.345,1.36],[0.001,0.01]]

# Initialize lists to store data
intensitydata = []
xvar_all=[]
yvar_all=[]
peaklocs = {}
peakamps = {}

bfields, bfield_list = pf.get_b_fields(file_paths)

# Read data from each file
for i, (bfield, file_path) in enumerate(bfield_list):
    # Read wavelength and intensity data from file
    energy, intensity, max_intensity = pf.read_spectrum(file_path, umBlaze = 0, normalize = 1, xEnergy = 1)
    # convert into data format for heatmap and apply settings
    yvar, zvar = pf.get_heatmap_data(energy, intensity, plot_with_mask = plot_with_mask, min_energy = min_energy, max_energy = max_energy, apply_smooth = applySmooth)
    yvar_all.append(yvar)  
    intensitydata.append(zvar)

    # Fit peaks
    x_data = yvar
    tofit = zvar
    
    if doFit == 1:
        fit = minimize(pf.diffL,p0,args=(x_data,tofit),bounds=bounds).get('x')
        fitpeaks = []
        fitamps = []
    
        for j in range(len(p0)//3):
            fitpeaks.append(fit[3*j+1])
            fitamps.append(fit[3*j])
        peaklocs['%s' % bfield] = fitpeaks
        peakamps['%s' % bfield] = fitamps
    
        # Plot each trace with fit
        if plotTraces == 1:
            plt.figure(dpi=400)
            plt.title("%s T" % bfields[i])
            plt.plot(x_data, tofit)
            plt.plot(x_data, pf.Lorentz(fit, x_data))
            plt.show()   

pf.plot_heatmap(bfields, yvar_all, intensitydata, plotTitle, cmap=cmap)
    
#%% Plot peak locations vs bfield
colors = ["r", "g", "b", "m", "c", "y"]

bypeak = []
for j in range(5):
    bypeak.append([])

for i in range(5):
    for bfield in bfields:
        locs = peaklocs["%s" % bfield]
        bypeak[i].append(locs[i])

pquad = [-3, 0, 1.4]

for i in range(len(peaklocs['0.0'])):
    peakNum = i+1
    quadfit = minimize(pf.diffquad, pquad, args=(bfields, bypeak[i])).get('x')
    a = round(quadfit[0], 5)
    b = round(quadfit[1], 5)
    c = round(quadfit[2], 2)
    quadEqn = "$y_%s$ = %s$x^2$ + %sx + %s" % (peakNum, a, b, c)
    plt.figure(dpi=400)
    plt.plot(bfields, bypeak[i], "r.")
    plt.title("Peak # %s, %s" % (peakNum, quadEqn)) 
    plt.plot(bfields, pf.quadratic(quadfit, bfields))
    plt.show()

#%% Get all values of peak 3
peak3Positions = []
for i, field in enumerate(peaklocs):
    peak3Positions.append(peaklocs[field][2])
print(max(peak3Positions)-min(peak3Positions))
# %%
