'''
Runs all of the TRR processing scripts and generates a report.

Noise removal step does not work if run in an interactive window
in the environment.

V1.0 created 9/04/2025 by Piper Aislinn,
Currently being updated.

Version 2.0
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #
from pathlib import Path
import sys

# Add code directory to system path to import custom modules.
THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))
from processing_package import ProcessingState

if __name__ == "__main__":

    experiment = ProcessingState(experiment_name = expt_date, 
                                   steps_list=['noise corrected',
                                               'normalized', 
                                               'processed', 
                                               'FFT'])
    
    # Generate outline for markdown report
    report_file = experiment.generate_report_skeleton()

    # --- Step 1: Remove noise --- #
    experiment.run_noise_remover()
        
    # --- Step 2: Processing (normalize and subtract decay) --- #
    
    # experiment.initialize_next_step()
    # experiment.normalize_data()
    # experiment.subtract_decay()

# # %%

# %%
