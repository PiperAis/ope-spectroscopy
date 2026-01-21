"""
Docstring for processing_state.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pathlib import Path
import re
import sys
import xarray as xr

THIS_DIR = Path(__file__).parent
sys.path.append(str(THIS_DIR))

import processing_package as pp
import config
from . import exp_decay

class ProcessingState:
    def __init__(self, 
                 experiment_name : str,
                 steps_list: list[str]):
        """
        :param experiment_name: name of folder that contains raw data 
        (usually a date)
        :type experiment_name: str
        :param steps_list: a list of the processing steps to be
        executed, these are used as headers for the data report when 
        generated and help create file structure and name plots.
        :type steps_list: list[str]

        This class holds directory paths for loading and saving data 
        and plots, as well as tracking the current data-processing 
        step to make plot labeling and report generation smoother.
        """
        
        self.steps_list = steps_list
        
        self.experiment_name = experiment_name
        if len(self.experiment_name) == 0:
            raise ValueError("Experiment name parameter is required to " \
            "accurately locate data.")

        # Initialize properties
        self.__step_number = 0
        self.__current_step = self.__steps_list[0]
        self.raw_data_dir = config.DATA_DIR / experiment_name
        self.save_dir = config.PROCESSED_DATA_DIR / experiment_name
        self.load_dir = self.raw_data_dir
        self.reports_dir = config.REPORTS_DIR
        self.plots_dir = config.PLOTS_DIR / experiment_name

        self.check_directory_existence()
    
    @property
    def steps_list(self):
        return self.__steps_list

    @steps_list.setter
    def steps_list(self, proposed_list):
        if len(proposed_list):
            self.__steps_list = proposed_list
        else:
            raise AttributeError("Steps list must have at least one element.")

    def check_directory_existence(self):
        """
        Checks whether directories for loading and saving data and 
        plots exist, and creates them if they do not.
        """
        if not self.raw_data_dir.exists():
            raise FileNotFoundError("Project folder does not contain the proper" \
            " structure to locate raw data or raw data is missing.")
        
        if not self.save_dir.exists():
            try:
                self.save_dir.mkdir()
            except FileNotFoundError:
                print("Folder for processed TRR data does not exist. Creating " \
                "TRR/processed directory.")
                self.save_dir.mkdir(parents = True)

        if not self.reports_dir.exists():
            try:
                self.reports_dir.mkdir()
            except FileNotFoundError:
                print("Issue with file structure. Check that TRR folder exists " \
                "under project root directory.")
        
        if not self.plots_dir.exists():
            self.plots_dir.mkdir()
        
        return None

    def initialize_next_step(self):
        self.__step_number += 1
        self.__current_step = self.steps_list[self.__step_number]

        previous_save_dir = self.save_dir
        self.load_dir = previous_save_dir

        self.check_directory_existence()

        return f'Initialized step: {self.__current_step}'

    def get_file_names(self, 
                       dir : pathlib.PurePath, 
                       condition : str = ''
                       ) -> list[str]:
        """
        :param dir: Directory containing files to be listed
        :type dir: pathlib.PurePath
        :param condition: Conditional for filtering file types.
        [f for f in dir.iterdir() if condition]
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
    
    def generate_report_skeleton(self, 
                                 diagnostic_mode : bool = False
                                 ) -> pathlib.PurePath:
        """
        Docstring for generate_report_skeleton
        
        :param diagnostic_mode: Set to True to print information about 
        report generation.
        :type diagnostic_mode: bool
        :return: Path to the .md file of the report.
        :rtype: PurePath
        """
        report_file_name = f'{self.raw_data_dir.name}-report.md'
        self.report_file = self.reports_dir / report_file_name
        report_file = self.report_file

        self.report_file.touch(exist_ok = True)

        file_names_list = self.get_file_names(
                                            self.raw_data_dir, 
                                            "not f.__contains__('.'')")

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
            this_moment = datetime.datetime.now().replace(microsecond=0)
            print(f"Report skeleton created at {this_moment} with " \
                  f"{len(file_names_list)} files.")

        return report_file
    
    def update_report(self, filename : str, 
                      caption : str = '', 
                      include_plot : bool = True):
        """
        Docstring for update_report
    
        :param filename: identifies the dataset to place plots and text 
        under correct heading in the report skeleton.
        :type filename: str
        :param caption: Text to include in report section.
        :type caption: str
        :param include_plot: Set to False if not including a plot in 
        this section.
        :type include_plot: bool
        """
        content = self.report_file.read_text()
        
        section = self.__current_step
        pattern = rf"(# {re.escape(filename)}[\s\S]*?## " \
                     rf"{re.escape(section)}\n)(\(placeholder\))"

        if include_plot:
            plot_path = str(self.plot_rel_path)
            replacement = rf"\1![]({plot_path})"
        else:
            replacement = ''
        
        if caption:
            replacement += f'\n\n*{caption}'
        
        new_content, count = re.subn(pattern, replacement, content, count=1)

        if count == 0:
            raise ValueError(
                f"Placeholder not found for {filename} / {section}"
                )

        with self.report_file.open(mode = 'a') as f:
            f.write(new_content)

        return None
    
    def run_noise_removal_process(self):
        filepaths_list = [f for f in self.load_dir.iterdir() 
                          if not str(f).__contains__(".")]

        for filepath in filepaths_list:
            filename = filepath.name
            set_number = filename.split('-')[0]
            plot_save_path = (self.plots_dir / 
                              f'{set_number}-{self.__current_step}.png')

            data_array = pp.prepare_data_array(filepath)

            data_array_original = data_array.copy()

            # Create interactive plot
            fig, ax = plt.subplots(figsize=(10, 5))
            remover = pp.NoiseRemover(data_array, data_array_original, ax, fig)
            fig.canvas.mpl_connect("button_press_event", remover.onclick)
            remover.update_plot()
            pp.add_noise_removal_buttons(fig, ax, remover)
            
            plt.show()

            final_selected_points = remover.selected_points

            # Plot results of noise removal
            plt.figure(figsize=(10, 5))
            plt.plot(data_array_original.time, data_array_original, 
                        label="Original Data", linestyle="dotted", alpha=0.5)
            plt.plot(data_array.time, data_array, 
                        label="Cleaned Data", linewidth=1)
            plt.legend()
            plt.xlabel("Time (ps)")
            plt.ylabel("Reflectance (arb. units)")
            plt.title(f"{set_number} noise removal")
            plt.savefig(fname = str(plot_save_path))
            plt.show()
            
            # Save data array in format usable by xarray module for next 
            # processing stage
            save_path = self.save_dir / f'{set_number}.nc'   

            # Create dataset to save raw and processed data in one object
            dataset = xr.Dataset({'raw' : data_array_original, 
                                  self.__current_step : data_array
                                  })
            dataset.attrs = data_array_original.attrs
    
            try:
                dataset.to_netcdf(path = str(save_path))
            except Exception:
                print("Error saving DataArray to NetCDF. " \
                      "Check if path length is too long.")
                print(f"save_path was: {str(save_path)}")   

            # Get relative path from report to plot
            # Kind of a hacky way of finding the plot path relative to the
            # report path. Perhaps can improve later.
            self.plot_rel_path = f"..{str(plot_save_path).split(r"TRR")[-1]}"

            self.update_report(filename)
        return

    def normalize_data(self):
        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]
        save_dir = self.save_dir

        for filepath in filepaths_list:
            filename = filepath.name
            dataset = xr.open_dataset(filepath)

            # Load dataset into memory and close file handle so we can 
            # overwrite safely
            dataset.load()
            dataset.close()

            # Get previous step name and load data
            prev_step = self.steps_list[self.__step_number - 1]
            data_array = dataset[prev_step].copy()

            # Check for existing values for scale factor and r0, 
            # and set to 1 if missing.
            if ('scale_factor' not in data_array.attrs and 
                'scale_factor_value' not in data_array.attrs):
                print("Scale factor not included in file name, setting to 1.")
                data_array.attrs['scale_factor_value'] = 1
            if ('rzero' not in data_array.attrs and 
                'r_zero' not in data_array.attrs):
                print("R_0 not included in file name, setting to 1.")
                data_array.attrs['rzero'] = 1
            
            data_array = pp.subtract_background(data_array, threshold = -2)
            data_array = pp.divide_out_factors(data_array)

            # Add the processed data to the dataset
            dataset[self.__current_step] = data_array

            # Save using mode = a to append new data to the dataset file.
            dataset.to_netcdf(path = save_dir / filename, mode = 'a')
        return None
    
    def subtract_decay(self):
        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]

        for filepath in filepaths_list:
            dataset = xr.open_dataset(filepath)
            set_number = filepath.name.split(".nc")[0]
            plot_save_path = (self.plots_dir / 
                              f"{set_number}-{self.__current_step}.png")
            # Load dataset into memory so .values is available, then close file 
            # to allow overwrite
            dataset.load()
            dataset.close()

            try:
                data_array = dataset['normalized'].copy()
            except:
                print("No variable named 'normalized' in the dataset. " \
                "Check that your processed data saved correctly during the " \
                "normalization step. Canceling subtraction step.")
                return None

            data_array = data_array.sel(
                time=~data_array.get_index("time").duplicated())

            # Keep a copy of the original values to restore masked regions later
            original_values = data_array.values.copy()

            # Mask the data 
            min_time = data_array.time.values[data_array.argmax()]
            max_time = data_array.time.values.max()

            masked_da = data_array.copy()
            masked_da = masked_da.where(data_array.time >= min_time)
            masked_da = masked_da.where(data_array.time <= max_time)

            masked_values = masked_da.values.copy()

            # Identify masked positions/indices
            mask = np.isnan(masked_values)

            # Get an array of the masked values and times with the NaNs dropped
            masked_values = masked_values[~mask]
            masked_times = masked_da.time.values[~mask]

            # Fit exponential decay to masked data, or subtract data avg if fit fails.
            initial_guess = [masked_values[0], 0.01, np.mean(masked_values)]

            try:
                params, _ = pp.curve_fit(
                    exp_decay, masked_da.time, masked_da.values, 
                    p0=initial_guess, nan_policy='omit')
                
                fit_da = xr.DataArray(
                    exp_decay(data_array.time.values, *params),
                    dims = data_array.dims,
                    coords = data_array.coords, 
                    name = 'fit')
                
                # Subtract fit from the original data, unmasked.
                processed_masked = xr.DataArray(
                    original_values - fit_da.values,
                    dims = data_array.dims,
                    coords = data_array.coords,
                    name = 'subtracted wonk'
                )

                # Reapply mask for times before t0.
                fitted_data_array = processed_masked.where(
                    data_array.time.values >= min_time)
                subtracted_data = fitted_data_array.values.copy()

                time_constant = params[0]

            except RuntimeError:
                print("Curve fit was unsuccessful. Subtracting DC component instead.")
                subtracted_data = masked_values - np.mean(masked_values)
                time_constant = 0

            # Create data array with subtracted data
            processed_data = xr.DataArray(
                subtracted_data,
                dims = data_array.dims,
                coords = data_array.coords,
                attrs = data_array.attrs,
                name = 'processed'
            )

            plt.figure(figsize=(10, 5))
            plt.plot(data_array.time, processed_data)
            plt.xlabel("Time (ps)")
            plt.ylabel("Differential reflectance")
            plt.title(f"{set_number} Decay Subtracted")
            plt.savefig(fname = f"{plot_save_path}")
            plt.show()

            self.update_report(filename)
            
            # Add time constant of subtracted decay to attributes
            processed_data.attrs['time_constant'] = time_constant

            # Add subtracted data to dataset
            dataset['subtracted'] = processed_data
            dataset.attrs['time_constant'] = time_constant

            # Generate save path
            filename = filepath.name
            save_path = self.save_dir / filename

            # Save dataset
            dataset.to_netcdf(path = str(save_path), mode = 'a')
        
        return
