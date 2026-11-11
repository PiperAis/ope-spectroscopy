"""
Analysis package for TRR and steady-state optical spectroscopy data.

Installed editable (pip install -e); projects import it rather than copying it.
The package provides capability; the project provides context — all paths and
parameters come from a project-level config.yaml, never hardcoded here.

Side effect on import (Windows only): sets HDF5_PLUGIN_PATH, if not already
set, to this package's folder, which holds no HDF5 plugins. The netCDF4 wheel
otherwise points HDF5 at its bundled filter plugins (zstd, blosc, bzip2, …),
and HDF5 then probes those DLLs for every variable it opens, about 40 ms each,
so loading a processed PL .nc (~30 variables) took ~1.6 s. Files using only
built-in compression (zlib, as xarray writes) are unaffected; a .nc compressed
with one of those plugin filters will fail to open. Set HDF5_PLUGIN_PATH
yourself before importing opespec to keep the plugins.
"""

import os as _os
import sys as _sys

if _sys.platform == 'win32':
    _os.environ.setdefault('HDF5_PLUGIN_PATH', _os.path.dirname(_os.path.abspath(__file__)))

from .fitting_functions import *
from .trr_dataset import *
from .noise_remover import *
from .fft_functions import *
from .processing_state import *
from .steady_state import *
from .pl_sweep import *
from .utilities import *