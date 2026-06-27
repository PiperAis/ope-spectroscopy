'''
Runs all of the TRR processing scripts and generates a report.

Noise removal step does not work if run in an interactive environment.
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #
import argparse
from pathlib import Path

# ROOT_DIR = Path(__file__).parent.parent
from processing_package import ProcessingState
from processing_package.utilities import load_config


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TRR processing pipeline.")
    parser.add_argument('--project', required=True,
                        help="Project folder name")
    args = parser.parse_args()

    cfg = load_config(args.project)

    experiment = ProcessingState(experiment_name = expt_date,
                                   steps_list=['noise corrected',
                                               'normalized',
                                               'processed',
                                               'FFT'],
                                   project_config = cfg)

    # Generate outline for markdown report
    report_file = experiment.generate_report_skeleton()

    # --- Step 1: Remove noise --- #
    experiment.run_noise_remover()
    experiment.normalize_data()
    experiment.subtract_decay()
    experiment.da_fourier_transform()
# # %%

# %%
