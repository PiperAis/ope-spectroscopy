'''
Config file for the project. Creates global variables and path names
for processing scripts to use.

Assumes code folder and TRR folder (where all files related to TRR
measurements are kept) are both in the root directory of the project.

'''
from pathlib import Path

THIS_DIR = Path(__file__).parent

TRR_DIR = THIS_DIR.parent / "TRR"

DATA_DIR = TRR_DIR / "raw_data"

PROCESSED_DATA_DIR = TRR_DIR / "processed"

REPORTS_DIR = TRR_DIR / "reports"

PLOTS_DIR = REPORTS_DIR / "plots"