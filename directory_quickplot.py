"""

"""
#%%
from os.path import dirname, join, isfile, basename

import numpy as np
import matplotlib.pyplot as plt
import glob
import pathlib
import sys

# Find code directory relative to our directory to import correct
# version of pipersfunctions.
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

from pipersfunctions import quick_plot, read_spectrum, mask

# Plot all in directory

dir_relative_path = r"..\PL\raw_data\2025-04-30\D8-bsweep"
directory = join(THIS_DIR, dir_relative_path)

for file in pathlib.Path(directory).glob('*'):
    if isfile(file):
        filename = basename(file)
        x, y, max = read_spectrum(file)
        x, y = mask(x, y, 1.25, 1.41)
        quick_plot(title = f'{filename}', x_var = x, y_var = y, x_label = 'Energy (ev)', y_label = 'PL (arb units)')
# %%
