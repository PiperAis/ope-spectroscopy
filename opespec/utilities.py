"""
Shared helpers used across data types (TRR and steady-state).

- load_config: read a project config.yaml and resolve all relative paths
- Filename parsers: get_field, get_temperature, get_sample_name, get_spot,
  get_scale_factor, get_rzero
- get_metadata: merges filename parsers with an optional folder-level
  metadata.txt (key: value lines) for parameters constant across a folder.
  Filename values win; a conflict raises a warning.
- File helpers: select_files, get_file_paths, get_file_names

Keep this data-type-agnostic — anything specific to one measurement type
belongs in that type's module. Filename format:
##-flake[sample]-[temp]K-[bfield]T-[scale]SF-[R0]mV-[tokens].
"""

from functools import lru_cache
from pathlib import Path
import re
import warnings
import yaml
import xarray as xr

_ROOT_DIR = Path(__file__).parent.parent
_SETTINGS_KEYS = (
    'projects_dir', 'vault_dir',
)
_DATA_PATH_KEYS = (
    'trr_dir', 'processed_trr_dir', 'reports_dir',
    'pl_dir', 'processed_pl_dir',
)


def _load_settings(settings_path) -> dict:
    """Load machine_settings.yaml and store paths."""
    settings_path = Path(settings_path).resolve()
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    for key in _SETTINGS_KEYS:
        if key in settings:
            settings[key] = Path(settings[key]).resolve()
    return settings

def load_config(project_name: str, settings_path: Path = _ROOT_DIR/'machine_settings.yaml') -> dict:
    """
    Load config.yaml for project (or use default config if project doesn't have its own) 
    and resolve all relative paths to absolute paths.
    """
    settings = _load_settings(settings_path)
    projects_dir = Path(settings['projects_dir'])
    vault_dir = settings.get('vault_dir')

    if not projects_dir.exists():
        warnings.warn(f"projects_dir does not exist: {projects_dir}. Check machine_settings.yaml.")

    config_path = (projects_dir / project_name / 'config.yaml').resolve()
    if not config_path.exists():
        config_path = (_ROOT_DIR / 'config.yaml').resolve()
        print("Project config not found, using default config.")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for key in _DATA_PATH_KEYS:
        if key in cfg:
            cfg[key] = (projects_dir / project_name / cfg[key]).resolve()
            if not cfg[key].exists():
                warnings.warn(f"Config path does not exist — {key}: {cfg[key]}")
    if vault_dir:
        cfg['vault_root'] = Path(vault_dir)
        cfg['project_vault'] = Path(vault_dir) / f'[{project_name}]'

    return cfg

#----File management----#

def select_files(initial_dir: Path) -> list[Path]:
    """Open a file browser in initial_dir; return selected CSV paths."""
    import tkinter as tk
    from tkinter import filedialog

    if not initial_dir.exists():
        raise FileNotFoundError(f"PL directory not found: {initial_dir}")
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    chosen = filedialog.askopenfilenames(
        title="Select PL data files to fit",
        initialdir=str(initial_dir),
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        parent=root,
    )
    root.destroy()
    return [Path(p) for p in chosen]

def select_directory(initial_dir: Path) -> Path | None:
    """Open a directory browser in initial_dir; return the chosen directory, or None."""
    import tkinter as tk
    from tkinter import filedialog

    if not initial_dir.exists():
        raise FileNotFoundError(f"PL directory not found: {initial_dir}")

    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    chosen = filedialog.askdirectory(
        title="Select directory of PL data files",
        initialdir=str(initial_dir),
        parent=root,
    )
    root.destroy()
    return Path(chosen) if chosen else None

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

#----Parsing Filenames----#

def get_field(filename : str | Path) -> float | None:
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
        warnings.warn(f"Could not parse magnetic field from filename: {filename}")
        return None

def get_sample_name(filename : str | Path, default='unnamed sample') -> str:
    """ Gets sample name from data filename, returning default
    ('unnamed sample') if not found.
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
        return default

def get_scale_factor(filename : str | Path, default=1) -> int:
    """
    Gets scale factor from data filename and returns it as an int.
    Returns default (1) if pattern is not found.
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
        return default

def get_rzero(filename : str | Path, default=1) -> float:
    """
    Gets r_0 from data filename and returns it as a float. Returns
    default (1) if pattern is not found.
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
        return default

def get_spot(filename: str, default='unknown spot') -> str:
    """
    Gets spot from filename.
    Assumes pattern like '-spot1-' after sample.
    """
    spot_match = re.search(r'spot.*-', filename)
    if spot_match:
        spot = spot_match.group(0)
        spot = spot.replace('-', '').replace('spot', 'Spot ')
        return spot
    return default

def get_temperature(filename: str | Path) -> float | None:
    """
    Gets temperature in Kelvin from filename (e.g. '1p6K' → 1.6, '10K' → 10.0).
    Returns None if not found.
    """
    match = re.search(r'-(\d+(p\d+)?)K\b', str(filename))
    if match:
        return float(match.group(1).replace('p', '.'))
    return None

#----Folder metadata----#

METADATA_FILENAME = 'metadata.txt'

# Defaults applied when neither the filename nor metadata.txt gives a value.
_METADATA_DEFAULTS = {
    'sample': 'unnamed sample',
    'spot': 'unknown spot',
    'field': None,
    'temperature': None,
    'scale_factor': 1,
    'rzero': 1,
}

@lru_cache
def _read_folder_metadata(folder: Path) -> dict:
    """Read folder/metadata.txt (key: value lines) once per folder; {} if absent."""
    path = folder / METADATA_FILENAME
    if not path.exists():
        return {}
    with open(path) as f:
        metadata = yaml.safe_load(f) or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} must contain 'key: value' lines.")
    return metadata

def get_metadata(filepath: str | Path) -> dict:
    """
    Merged metadata for a data file. Values parsed from the filename
    override those in metadata.txt in the file's folder (warning on
    conflict). Keys: sample, spot, field, temperature, scale_factor,
    rzero, plus any extra keys from metadata.txt.
    """
    filepath = Path(filepath)
    name = filepath.name
    folder_metadata = _read_folder_metadata(filepath.parent.resolve())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = get_field(name)
    from_filename = {
        'sample': get_sample_name(name, default=None),
        'spot': get_spot(name, default=None),
        'field': field,
        'temperature': get_temperature(name),
        'scale_factor': get_scale_factor(name, default=None),
        'rzero': get_rzero(name, default=None),
    }

    metadata = dict(folder_metadata)
    for key, value in from_filename.items():
        folder_value = folder_metadata.get(key)
        if value is not None:
            if folder_value is not None and folder_value != value:
                warnings.warn(f"{name}: filename {key}={value!r} overrides "
                              f"{METADATA_FILENAME} {key}={folder_value!r}")
            metadata[key] = value
        elif folder_value is None:
            metadata[key] = _METADATA_DEFAULTS[key]

    return metadata

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