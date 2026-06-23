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
import argparse
from pathlib import Path
import sys
import yaml

# Add code directory to system path to import custom modules.
THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))
from processing_package import ProcessingState


def load_config(config_path):
    """Load config.yaml and resolve any relative paths relative to the yaml file."""
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    for key in ('data_dir', 'processed_data_dir', 'reports_dir', 'pl_dir'):
        if key in cfg:
            cfg[key] = (base / cfg[key]).resolve()
    return cfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TRR processing pipeline.")
    parser.add_argument(
        '--config',
        default=str(THIS_DIR / 'config.yaml'),
        help="Path to project config.yaml (default: config.yaml next to this script)"
    )
    args, _ = parser.parse_known_args()

    cfg = load_config(args.config)

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

    # --- Step 2: Processing (normalize and subtract decay) --- #

    experiment.normalize_data()
    experiment.subtract_decay()
    experiment.da_fourier_transform()
# # %%

# %%
