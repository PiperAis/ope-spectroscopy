"""
Shared utility functions such as filename parsing.
"""

import re
import pathlib
from pathlib import Path
import xarray as xr

#----Parsing Filenames----#

def get_field(filename : str | pathlib.PurePath) -> float | None:
    """ 
    Gets magnetic field from data filename, returning it as a str or 
    None if not found. Assumes filename structure is (e.g.)
    ***-n2p6T-*** where n represents negative sign, p represents a
    period, and the field is separated from other information by '-'
    """
    # Search for the regex pattern denoting bfield
    match = re.search(r'-n?\d+(p\d+)?T', str(filename))

    # Convert filename formatting to float
    if match:
        bstring = match.group(0)
        bstring = bstring.replace('-', '')
        bstring = bstring.replace('p', '.')
        bstring = bstring.replace('n', '-')
        bstring = bstring.replace('T', '')
        bfield = float(bstring)
        return bfield
    else:
        return None

def get_sample_name(filename : str | pathlib.PurePath) -> str:
    """ Gets sample name from data filename, returning 'unnamed
    sample' if not found. 
    """
    # Search for the regex pattern denoting flake name
    match = re.search(r'flake.*-', str(filename))

    # Format sample name
    if match:
        sample_name = match.group(0)
        sample_name = sample_name.split('-')[0]
        sample_name = sample_name.replace('-', '')
        sample_name = sample_name.replace('flake', 'Flake ')
        return sample_name
    else:
        return 'unnamed sample'

def get_scale_factor(filename : str | pathlib.PurePath) -> int:
    """ 
    Gets scale factor from data filename and returns it as an int. 
    Returns 1 if pattern is not found. 
    """
    # Search for regex pattern denoting scale factor
    match = re.search(r'-\d+kSF', str(filename))

    # Format scale factor
    if match:
        scale_factor = match.group(0)
        scale_factor = scale_factor.replace('-', '')
        scale_factor = scale_factor.replace('SF', '')
        scale_factor = scale_factor.replace('k', '000')
        scale_factor = int(scale_factor)
        return scale_factor
    else:
        return 1

def get_rzero(filename : str | pathlib.PurePath) -> float:
    """ 
    Gets r_0 from data filename and returns it as a float. Returns 1 if 
    pattern is not found. 
    """
    # Search for regex pattern denoting r_0
    match = re.search(r'-\d+p?(\d+)?mV', str(filename))

    # Format value for return
    if match:
        rzero = match.group(0)
        rzero = rzero.replace('-', '')
        rzero = rzero.replace('mV', '')
        rzero = rzero.replace('p', '.')
        rzero = float(rzero)
        return rzero
    else:
        return 1
    
def get_spot(filename: str) -> str:
    """
    Gets spot from filename.
    Assumes pattern like '-spot1-' after sample.
    """
    spot_match = re.search(r'spot.*-', filename)
    if spot_match:
        spot = spot_match.group(0)
        spot = spot.replace('-', '').replace('spot', 'Spot ')
        return spot
    return 'unknown spot'

def get_file_names(dir : Path, 
                    condition : str = ''
                    ) -> list[str]:
    """
    Returns a list of the file names in dir in alphabetical order.

    :param dir: Directory containing files to be listed
    :type dir: pathlib.PurePath
    :param condition: Conditional for filtering file types.
    [f for f in dir.iterdir() if condition]
    :type condition: str
    """
    if condition:
        filepaths = [f for f in dir.iterdir() if condition]
    else:
        filepaths = [f for f in dir.iterdir()]

    filenames = [f.name for f in filepaths]
    filenames.sort()

    return filenames


#----xarray utilities----#

def mask_data_array(
        data_array : xr.DataArray, 
        min_time : int | float, 
        max_time : int | float
        ) -> xr.DataArray:
    """ 
    Returns a DataArray with datapoints outside of the given time
    limits replaced with NaN. 
    """
    condition = (data_array >= min_time) and (data_array <= max_time)
    new_array = data_array.where(condition).all()
    
    return xr.DataArray(new_array)