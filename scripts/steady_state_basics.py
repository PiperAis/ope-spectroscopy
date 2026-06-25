"""
Executable script for steady-state processing.

Rough draft, finishing touches will come during phase 5!
"""
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data
#%%
expt_date = "test"

# ---- Import modules ---- #
import argparse
from pathlib import Path
import yaml

# Add code directory to system path to import custom modules.
ROOT_DIR = Path(__file__).parent.parent
from processing_package import steady_state

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

    parser = argparse.ArgumentParser(description="Run PL processing pipeline.")
    parser.add_argument(
        '--project',
        default=None,
        help="Name of the project directory containing config.yaml"
    )
    args, _ = parser.parse_known_args()

    if args.project:
        config_path = ROOT_DIR.parent / args.project / 'config.yaml'
    else:
        config_path = ROOT_DIR.parent / '07_CrSBr_Cryo4' / 'config.yaml'

    cfg = load_config(config_path)

    experiment_date = '2026-03-03'
    data_path = Path(cfg['pl_dir']) / experiment_date
    results = steady_state.process_pl_directory_gaussian(data_path)
    for res in results:
        print(res)
# %%
