'''
Helper functions for generating data reports for transient reflectance 
experiments.

Created 9/04/2025 by Piper Aislinn

Version 1.0
'''
#%%
import sys
import os
import datetime
import re

from os.path import join, dirname

# Find code directory relative to our directory to import correct 
# version of custom modules.
THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)

import config

data_dir = config.DATA_DIR
processed_data_dir = config.PROCESSED_DATA_DIR
plots_dir = config.PLOTS_DIR
reports_dir = config.REPORTS_DIR

def generate_skeleton(raw_data_dir : str, steps : list[str] = [
    "Raw + noise corrected",
    "Processed",
    "Fourier transform"
    ]):
    ''' Creates a markdown file with outline for a data report. Returns
    path to the report file. '''

    files = [f for f in os.listdir(raw_data_dir) if not f.__contains__(".")]
    files.sort() 

    report_file = os.path.join(reports_dir, f"{os.path.basename(raw_data_dir)}-report.md")

    with open(report_file, "w") as f:
        f.write("# Data Processing Report\n\n")

        f.write(f'Data collected {os.path.basename(raw_data_dir)}\n')

        for fname in files:
            f.write(f"# {fname}\n\n")
            for step in steps:
                f.write(f"## {step}\n")
                f.write("(placeholder)\n\n")
            f.write("\n")

    print(f"Report skeleton created at {datetime.datetime.now().replace(microsecond=0)} with {len(files)} files.")
    
    return report_file


def update_report(report_file : str, filename : str, section : str, plot_path: str, caption : str = "") -> None:
    """
    Replace a placeholder in the Markdown report with an image.
    
    Args:
        report file (str): path to report markdown file
        filename (str): e.g. "file1.dat"
        section (str): the subsection title, e.g. "Raw + noise corrected"
        plot_path (str): path to the plot image (relative to the .md file)
        caption (str): optional text under the plot
    """
    with open(report_file, "r") as f:
        content = f.read()

    # Regex pattern: find the section under the filename
    pattern = rf"(# {re.escape(filename)}[\s\S]*?## {re.escape(section)}\n)(\(placeholder\))"

    if plot_path:
        plot_path = plot_path.replace("\\", "/")
        replacement = rf"\1![]({plot_path})"
    else:
        replacement = ""
    if caption:
        replacement += f"\n\n*{caption}*"

    # Replace only the first placeholder occurrence
    new_content, count = re.subn(pattern, replacement, content, count=1)

    if count == 0:
        raise ValueError(f"Placeholder not found for {filename} / {section}")

    with open(report_file, "w") as f:
        f.write(new_content)

    return

def initialize_next_step(state : dict, steps_list : list) -> dict:
    """ Sets paths for loading and saving data and images. 
    Takes state dictionary which should be initialized as: \n
    state = {'step_number' : 0, 
         'load_dir' : raw_data_dir, 
         'save_dir' : '', 
         'plot_save_dir' : ''}
    """
    new_state = {
                # 'processed_data_dir' : state['processed_data_dir'],
                 'plots_dir' : state['plots_dir'],
                 'save_dir' : state['save_dir']}

    # Get name of current step
    step_number = state['step_number']
    step_number += 1
    current_step = steps_list[step_number]
    new_state['step_number'] = step_number
    # Set saved data from previous step as load for next step
    new_state['load_dir'] = state['save_dir']

    # Set plot save dir as subfolder w/ step name in plots dir
    plot_save_dir = join(state['plots_dir'], current_step)
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)
    new_state['plot_save_dir'] = plot_save_dir

    return new_state


# %%
