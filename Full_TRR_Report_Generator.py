'''
Runs all of the TRR processing scripts and generates a report.

Noise removal step does not work if run in an interactive window
in the environment.

Created 9/04/2025 by Piper Aislinn

Version 1.0
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #

import numpy as np
import pathlib
from pathlib import Path
import sys
import xarray as xr

# Add code directory to system path to import custom modules.

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))

# Import custom modules

import processing_package as pp

if __name__ == "__main__":

    experiment = pp.ProcessingState(experiment_name = 'test', 
                                   steps_list=['noise corrected', 
                                               'processed', 
                                               'fourier transform'])
    
    # Generate outline for markdown report
    report_file = experiment.generate_report_skeleton()

    # --- Step 1: Remove noise --- #
    experiment.run_noise_removal_process()
        
    # --- Step 2: Processing (normalize and subtract decay) --- #
    
    experiment.initialize_next_step()
    experiment.normalize_data()

#         subtraction_results = pp.subtract_decay(filepath, str(plots_dir), save_dir, str(report_file))

# # %%
