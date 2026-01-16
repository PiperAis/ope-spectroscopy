'''
Config file for the project. Creates global variables and path names
for processing scripts to use.

'''
import os
from os.path import dirname, join

THIS_DIR = dirname(__file__)

raw_data_dir_relative_path = r"..\TRR\raw_data"
DATA_DIR = join(THIS_DIR, raw_data_dir_relative_path)

processed_data_dir = r"..\TRR\processed"
PROCESSED_DATA_DIR = join(THIS_DIR, processed_data_dir)

plots_dir_relative_path = r'..\TRR\plots'
PLOTS_DIR = join(THIS_DIR, plots_dir_relative_path)

reports_dir_rel_path = r"..\TRR\reports"
REPORTS_DIR = join(THIS_DIR, reports_dir_rel_path)