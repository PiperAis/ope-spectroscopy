"""
Docstring for Untitled-1
"""

import pathlib
from pathlib import Path
import sys

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))

import processing_package as pp
from processing_package import config

# Import custom modules

import processing_package as pp
import config

class ProcessingState:
    def __init__(self, 
                 experiment_name : str,
                 steps_list: list[str]):
        """
        :param experiment_name: name of folder that contains raw data (usually a date)
        :type experiment_name: str
        :param steps_list: a list of the processing steps to be executed, these
        are used as headers for the data report when generated and help create
        file structure and name plots.
        :type steps_list: list[str]

        This class holds directory paths for loading and saving data and plots, as well
        as tracking the current data-processing step to make plot labeling and report
        generation smoother.
        """
        
        self.steps_list = steps_list
        if len(steps_list) == 0:
            raise ValueError("Steps list must have at least one step.")
        
        self.experiment_name = experiment_name
        if len(self.experiment_name) == 0:
            raise ValueError("Experiment name parameter is required to accurately locate data.")

        # Initialize properties
        self.step_number = 0
        self.current_step = self.steps_list[0]
        self.raw_data_dir = config.DATA_DIR / experiment_name
        self.save_dir = config.PROCESSED_DATA_DIR / experiment_name
        self.load_dir = self.raw_data_dir
        self.reports_dir = config.REPORTS_DIR
        self.plots_dir = config.PLOTS_DIR / experiment_name

        self.check_directory_existence()

    def check_directory_existence(self):
        """
        Checks whether directories for loading and saving data and plots exist,
        and creates them if they do not.
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError("Project folder does not contain the proper structure to locate raw data or raw data is missing.")
        
        if not self.save_dir.exists():
            try:
                self.save_dir.mkdir()
            except FileNotFoundError:
                print("Folder for processed TRR data does not exist. Creating TRR/processed directory.")
                self.save_dir.mkdir(parents = True)

        if not self.reports_dir.exists():
            try:
                self.reports_dir.mkdir()
            except FileNotFoundError:
                print("Issue with file structure. Check that TRR folder exists under project root directory.")
        
        if not self.plots_dir.exists():
            self.plots_dir.mkdir()
        
        return None

    def next_step(self):
        self.step_number += 1
        self.current_step = self.steps_list[self.step_number]

        previous_save_dir = self.save_dir
        self.load_dir = previous_save_dir

        return f'Current step: {self.current_step}'

    def get_file_names(self, dir : pathlib.PurePath, condition : str) -> list[str]:
        """
        :param dir: Directory containing files to be listed
        :type dir: pathlib.PurePath
        :param condition: Conditional for filtering file types. 
        Start with 'if' then write out conditional statement.
        :type condition: str

        Returns a list of the file names in dir in alphabetical order.
        """
        if condition:
            filepaths = [f for f in dir.iterdir() if condition] #type: ignore
        else:
            filepaths = [f for f in dir.iterdir()] #type: ignore

        filenames = [f.name for f in filepaths]
        filenames.sort()

        return filenames
    
    def generate_report_skeleton(self, diagnostic_mode : bool = False) -> pathlib.PurePath:
        report_file_name = f'{self.raw_data_dir.name}-report.md'
        self.report_file = self.reports_dir / report_file_name
        report_file = self.report_file

        self.report_file.touch(exist_ok = True)

        file_names_list = self.get_file_names(self.raw_data_dir, "not f.__contains__('.'')")

        with report_file.open(mode='a') as f:
            f.write(f"# Data Processing Report for {self.experiment_name}\n\n")

            for fname in file_names_list:
                f.write(f"# {fname}\n\n")

                for step in self.steps_list:
                    f.write(f"## {step}\n")
                    f.write("(placeholder)\n\n")

                f.write("\n")

        if diagnostic_mode == True:
            import datetime   
            print(f"Report skeleton created at {datetime.datetime.now().replace(microsecond=0)} with {len(file_names_list)} files.")

        return report_file
    
    def update_report(self, caption : str):

        return NotImplemented


    
