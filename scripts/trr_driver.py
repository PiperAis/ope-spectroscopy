'''
Runs all TRR processing steps and generates a report.

Noise removal step does not work if run in an interactive environment.

Usage:
    python scripts/trr_driver.py --project <project_folder_name>
    python scripts/trr_driver.py --project <project_folder_name> --sync-obsidian
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #
import argparse
import shutil

from opespec import ProcessingState
from opespec.utilities import load_config
from opespec.obsidian_integration import generate_dataset_notes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TRR processing pipeline.")
    parser.add_argument('--project', required=True,
                        help="Project folder name")
    parser.add_argument('--sync-obsidian', action='store_true',
                        help="Copy report to Obsidian vault and generate dataset notes")
    args = parser.parse_args()

    cfg = load_config(args.project)

    experiment = ProcessingState(experiment_name=expt_date,
                                 steps_list=['noise corrected',
                                             'normalized',
                                             'processed',
                                             'FFT'],
                                 project_config=cfg)

    report_file = experiment.generate_report_skeleton()

    experiment.run_noise_remover()
    experiment.normalize_data()
    experiment.subtract_decay()
    experiment.da_fourier_transform()

    if args.sync_obsidian:
        shutil.copytree(cfg['reports_dir'] / expt_date,
                        cfg['project_vault'] / expt_date / 'report',
                        dirs_exist_ok=True)
        generate_dataset_notes(
            processed_data_dir=experiment.save_dir,
            vault_expt_dir=experiment.vault_dir,
            expt_name=expt_date,
        )

# %%
