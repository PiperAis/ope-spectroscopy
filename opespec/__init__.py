"""
Analysis package for TRR and steady-state optical spectroscopy data.

Installed editable (pip install -e); projects import it rather than copying it.
The package provides capability; the project provides context — all paths and
parameters come from a project-level config.yaml, never hardcoded here.
"""

from .fitting_functions import *
from .trr_dataset import *
from .noise_remover import *
from .fft_functions import *
from .processing_state import *
from .steady_state import *
from .utilities import *