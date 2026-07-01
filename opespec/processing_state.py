"""
ProcessingState: linear-pipeline orchestrator for TRR data.

Tracks processing steps, manages file flow between them, and builds the
per-step audit report. Steps run in sequence (noise removal → normalize →
subtract decay → FFT), each emitting a plot into the report.

DO NOT generalize this to steady-state. Steady-state analysis is interactive
fitting + cross-dataset comparison — a different shape that does not fit this
pipeline frame. Share helpers, not this orchestrator. (See CLAUDE.md.)

Paths come from the project config passed to __init__; this class hardcodes
no locations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pathlib
from pathlib import Path
import re
from scipy.optimize import curve_fit
import xarray as xr

from .trr_dataset import TRRDataset, subtract_background, divide_out_factors
from .fitting_functions import exp_decay
from . import utilities

class ProcessingState:
    def __init__(self,
                 experiment_name : str,
                 steps_list: list[str],
                 project_config: dict):
        """
        :param experiment_name: name of folder that contains raw data
        (usually a date)
        :type experiment_name: str
        :param steps_list: a list of the processing steps to be
        executed, these are used as headers for the data report when
        generated and help create file structure and name plots.
        :type steps_list: list[str]
        :param project_config: dict of resolved paths and settings,
        typically loaded from a project-level config.yaml. Required
        keys: trr_dir, processed_trr_dir, reports_dir.
        :type project_config: dict

        This class holds directory paths for loading and saving data
        and plots, as well as tracking the current data-processing
        step to make plot labeling and report generation smoother.
        """

        self.steps_list = steps_list

        self.experiment_name = experiment_name
        if len(self.experiment_name) == 0:
            raise ValueError("Experiment name parameter is required to " \
            "accurately locate data.")

        self.__step_number = 0
        self.__current_step = self.__steps_list[0]
        self.__previous_step = None

        self.raw_data_dir = Path(project_config['trr_dir']) / experiment_name
        self.save_dir = Path(project_config['processed_trr_dir']) / experiment_name
        self.reports_dir = Path(project_config['reports_dir']) / experiment_name
        self.vault_dir = Path(project_config['project_vault']) / experiment_name
        self.load_dir = self.save_dir
        self.plots_dir = self.reports_dir

        self.babyfresh = True
        
        self.datasets : list[xr.Dataset] = []

        self.check_directory_existence()
        self.prepare_files()
    
    @property
    def steps_list(self):
        return self.__steps_list

    @steps_list.setter
    def steps_list(self, proposed_list):
        if len(proposed_list) and all([type(proposed_list[i]) == str for i in range(len(proposed_list))]):
            self.__steps_list = proposed_list
        elif not len(proposed_list):
            raise AttributeError("Steps list must have at least one element.")
        else:
            raise TypeError("All steps must be strings.")
    
    @property
    def current_step(self):
        return self.steps_list[self.__step_number]
    
    @current_step.setter
    def current_step(self, step_name : str):
        step = step_name.replace('_', ' ').lower()
        if step in self.__steps_list:
            self.__current_step = step
            current_step_number = self.__steps_list.index(step)
            self.__step_number = current_step_number
            try:
                self.__previous_step = self.steps_list[current_step_number - 1]
            except IndexError:
                self.__previous_step = None
        else:
            raise ValueError("Step name not found in steps list.")
    
    @property
    def previous_step(self):
        step_num = self.__step_number - 1
        if step_num >= 0:
            previous_step = self.steps_list[step_num]
        else:
            previous_step = 'raw'
        return previous_step

    def __str__(self):
        return f"Experiment from {self.experiment_name}. Steps: {self.steps_list} You are on {self.current_step}. \n\
            Datasets: {self.datasets}."

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
        '''
        Updates variables that track the current and previous
        processing steps and sets the new load directory as the save
        directory from the previous step. Ensures directories exist. \n
        Returns a string that states which step was just initialized.
        '''
        # If this is the first step, initialize but don't advance
        if self.babyfresh:
            self.babyfresh = False
            return 'Now we\'re gettin started'
        
        # Otherwise, move forward along the step chain
        self.__step_number += 1
        self.__previous_step = self.__current_step
        self.__current_step = self.steps_list[self.__step_number]

        previous_save_dir = self.save_dir
        self.load_dir = previous_save_dir

        self.check_directory_existence()

        return f'Initialized step: {self.__current_step}'
    
    def get_plot_save_path(self, dataset : xr.Dataset, as_string = True)-> str | pathlib.PurePath:
        step = self.__current_step
        stepstr = step.replace(' ', '-')
        plot_save_path = (self.plots_dir /
                          f"{dataset.trrxr.set_number}-{stepstr}.png")
        if as_string:
            pathstr = str(plot_save_path)
            return pathstr
        else:
            return plot_save_path
    
    def generate_report_skeleton(self, 
                                 diagnostic_mode : bool = False
                                 ) -> pathlib.PurePath:
        """
        Generates a markdown file skeleton with placeholder headings for
        each dataset in the experiment folder and subheadings for each 
        processing step.
        
        :param diagnostic_mode: Set to True to print information about 
        report generation.
        :type diagnostic_mode: bool
        :return: Path to the .md file of the report.
        :rtype: PurePath
        """
        report_file_name = f'{self.experiment_name}-report.md'
        self.report_file = self.reports_dir / report_file_name

        file_names_list = utilities.get_file_names(
                                            dir = self.raw_data_dir, 
                                            condition = "not f.__contains__('.'')")

        self.report_file.write_text(f"# Data Processing Report for {self.experiment_name}\n\n")

        with self.report_file.open(mode='a') as f:
            for fname in file_names_list:
                f.write(f"# {fname}\n\n")

                for step in self.steps_list:
                    f.write(f"## {step}\n")
                    f.write("placeholder\n\n")

                f.write("\n")

        if diagnostic_mode == True:
            import datetime
            this_moment = datetime.datetime.now().replace(microsecond=0)
            print(f"Report skeleton created at {this_moment} with " \
                  f"{len(file_names_list)} files.")

        return self.report_file
    
    def prepare_files(self) -> None:
        '''
        Takes any TimeSpec files (no file extension) in the load dir
        and saves them as an xarray DataArray in a dataset, saving
        metadata from original filename in dataset.attrs, then saves
        the .nc files to the save directory.
        '''
        filepaths_list = [f for f in self.raw_data_dir.iterdir() 
                    if not str(f).__contains__(".")] # TimeSpec files lack a file extension
        
        for filepath in filepaths_list:
            dataset = TRRDataset.create_trr_dataset(filepath)
            try:
                dataset.trrxr.save(self.save_dir)
            except OSError as e:
                print("Error saving DataArray to NetCDF. " \
                      f"Path may be too long. ({e})")
            self.datasets.append(dataset)

        self.__previous_step = 'raw'

        return None
    
    def update_report(self, dataset : xr.Dataset, 
                      caption : str = '', 
                      include_plot : bool = True):
        """
        Updates the markdown report file for this experiment with plot
        generated by the current processing step and adds the caption
        if included.
    
        :param dataset: identifies the dataset to place plots and text 
        under correct heading in the report skeleton.
        :type filename: xr.Dataset
        :param caption: Text to include in report section.
        :type caption: str
        :param include_plot: Set to False if not including a plot in 
        this section.
        :type include_plot: bool
        """
        filename = dataset.attrs['data filename']
        section = self.__current_step
        replacement=''
        
        if include_plot:
            plot_path = self.get_plot_save_path(dataset, as_string=False)
            plot_rel_path = plot_path.name #type: ignore
            plot_path = str(plot_rel_path).replace("\\", "\\\\")
            replacement = rf"\1![]({plot_path})"
        else:
            replacement = 'No plot included.'
        
        if caption:
            replacement += f'\n\n*{caption}'
        
        # Open existing report text, find placeholders, and insert new content.
        content = self.report_file.read_text()
        pattern = rf"(# {re.escape(filename)}[\s\S]*?## {re.escape(section)}\n)placeholder"
        new_content, count = re.subn(pattern, replacement, content, count=1)

        if count == 0:
            raise ValueError(\
                f"Placeholder not found for {filename} / {section}")

        with self.report_file.open(mode = 'w') as f:
            f.write(new_content)

        return None

    def run_noise_remover(self):
        self.initialize_next_step()

        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]

        for filepath in filepaths_list:
            dataset = xr.load_dataset(filepath)
            dataset.trrxr.remove_noise(processor = self)
            self.update_report(dataset)
            dataset.trrxr.save(self.save_dir)

        return

    def normalize_data(self):
        self.initialize_next_step()
        try:
            assert self.__current_step == 'normalized'
        except Exception as e:
            raise AssertionError('Initialized normalization step but current step does not match.', e)
        
        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]

        for filepath in filepaths_list:
            dataset = xr.load_dataset(filepath)

            # Load data from the previous processing step
            data_array = dataset[self.__previous_step].copy()

            # Copy normalization metadata from dataset attrs to the data array
            # so divide_out_factors can find them.
            data_array.attrs['sf'] = dataset.trrxr.sf or 1
            data_array.attrs['rzero'] = dataset.trrxr.rzero or 1

            data_array = subtract_background(data_array, threshold = -2)
            data_array = divide_out_factors(data_array)

            # Add the processed data to the dataset and save
            dataset.trrxr.add_array(self.__current_step, data_array)
            dataset.trrxr.save(self.save_dir)
        return None
    
    def subtract_decay(self):
        self.initialize_next_step()
        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]

        for filepath in filepaths_list:
            dataset = xr.load_dataset(filepath)
            dataset = dataset.isel(time=~dataset.get_index("time").duplicated())
            dataset.trrxr._verify_attributes()

            data_array = dataset[self.__previous_step].copy()

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
                params, _ = curve_fit(
                    exp_decay, masked_da.time, masked_da.values, 
                    p0=initial_guess, nan_policy='omit')
                time_constant = 1 / params[1]

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
                    name = 'subtracted temp'
                )

                # Reapply mask for times before t0.
                fitted_data_array = processed_masked.where(
                    data_array.time.values >= min_time)
                subtracted_data = fitted_data_array.values.copy()

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
            plt.title(f"{dataset.attrs['set number']} Decay Subtracted")
            plt.savefig(fname = self.get_plot_save_path(dataset))
            plt.clf()

            self.update_report(dataset) #type: ignore
            
            # Add time constant of subtracted decay to attributes
            processed_data.attrs['time_constant'] = time_constant

            # Add subtracted data to dataset
            dataset.trrxr.add_array(self.__current_step, processed_data)
            dataset.trrxr.save(self.save_dir)
        return

    def da_fourier_transform(self, apply_hanning_window : bool = True, max_frequency : float = 100):
        '''
        Fourier transforms data, divides frequencies by 1000 -
        expecting ps input this outputs GHz frequencies.
        
        :param self: Description
                '''   
        def uniform_time_grid(time_vals, y_vals):
                    """ Takes time values and returns a uniform time grid with the same or more points """
                    from scipy.interpolate import interp1d
                    num_points=len(y_vals)
                    t_uniform = np.linspace(time_vals.min(), time_vals.max(), num_points)
                    interp_func = interp1d(time_vals, y_vals, kind='linear')
                    y_uniform = interp_func(t_uniform)
                    
                    return t_uniform, y_uniform
        
        self.initialize_next_step()
        
        prev_save_dir = self.save_dir
        new_save_dir = prev_save_dir / 'FFT'    
        self.save_dir = new_save_dir
        self.check_directory_existence()
        
        filepaths_list = [f for f in self.load_dir.iterdir() if f.is_file()]

        for filepath in filepaths_list:
            dataset = xr.load_dataset(filepath)
            dataset = dataset.isel(time=~dataset.get_index("time").duplicated())

            data_array = dataset[self.__previous_step].copy()

            da = data_array.dropna('time')
            y_vals = da.values.copy()
            time_vals = da.time.values.copy()

            time_vals, y_vals = uniform_time_grid(time_vals, y_vals)
            if apply_hanning_window:
                window = np.hanning(len(y_vals))
                y_vals = y_vals * window
            
            dt = np.mean(np.diff(time_vals))
            fft_result = np.fft.fft(y_vals)
            fft_magnitude = np.abs(fft_result)
            freqs = np.fft.fftfreq(len(y_vals), d=dt)
            freqs = [f*1000 for f in freqs]

            fft_dataarray = xr.DataArray(data = fft_magnitude, coords = {'frequency' : freqs}, name = 'raw', attrs = dataset.attrs)

            freq_mask = (fft_dataarray.frequency >= 0) & (fft_dataarray.frequency <= max_frequency)
            masked_array = fft_dataarray.where(freq_mask, drop = True)
            # masked_array.plot.scatter()

            plt.figure(figsize=(10, 5))
            # plt.plot(freqs, fft_magnitude)
            plt.plot(masked_array.frequency, masked_array.values)
            plt.xlabel("Frequency (GHz)")
            plt.ylabel("FFT Magnitude")
            plt.title(f"{dataset.trrxr.set_number} Fourier Transform")
            plt.savefig(fname = self.get_plot_save_path(dataset))

            self.update_report(dataset) #type: ignore

            save_path = self.save_dir / f'{dataset.trrxr.set_number}-fft.nc'
            fft_dataarray.to_netcdf(save_path)

        return