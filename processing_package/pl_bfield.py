"""
Helper functions for processing PL vs. magnetic field (B-sweep) data.

Ported from pipersfunctions.py v1.3.
"""
from pathlib import Path
from . import utilities
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.constants as const

def get_file_paths(
                    dir : Path, 
                    condition : str = ''
                    ) -> list[Path]:
    """
    Returns a list of the file names in dir in alphabetical order.

    :param dir: Directory containing files to be listed
    :type dir: pathlib.PurePath
    :param condition: Conditional for filtering file types.
    [f for f in dir.iterdir() if condition]
    :type condition: str
    """
    if condition:
        filepaths = [f for f in dir.iterdir() if condition] #type: ignore
    else:
        filepaths = [f for f in dir.iterdir()] #type: ignore

    return filepaths

def bfield_contour_plot(data_directory : Path):
    
    bfield_file_list = get_file_paths(data_directory)
    spectra = []
    for filepath in sorted(bfield_file_list):
        df = pd.read_csv(filepath)                    # one spectrum
        field = utilities.get_field(filepath)          # reuse the existing helper
        da = xr.DataArray(
            df['Intensity'].values,
            coords={"wavelength": df['Wavelength'].values,
                    },
            dims=["wavelength"],
        )
        energy_values = const.h * const.c / (da['wavelength'].values * 1e-9) / const.e
        da = da.expand_dims(field=[field])              # tag this spectrum with its field
        da = da.assign_coords(
            energy=("wavelength", energy_values)
            )
        spectra.append(da)

    bfield_map = xr.concat(spectra, dim="field")       # stack into 2D
    bfield_map.plot.contourf(x="Magnetic Field (T)", y="Energy (eV)")    # contour map, axes auto-labeled
    plt.show()
