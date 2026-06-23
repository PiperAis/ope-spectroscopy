"""
Driver script for PL vs. B-field processing.

Generates a contour (heatmap) plot of PL intensity as a function of
energy and magnetic field. Optionally fits each spectrum with Lorentzians
and plots peak positions vs. field.

Data directory is read from config.yaml (pl_bfield_dir key).
Pass --project <folder name> to use a different project's config;
defaults to 07_CrSBr_Cryo4.

Usage:
    python PL_bfield_driver.py
    python PL_bfield_driver.py --project <project_folder_name>
"""

# ---- User settings ---- #

# Apply energy bounds to the spectra
plot_with_mask = True
min_energy = 1.3
max_energy = 1.35

# Apply Savitzky-Golay smoothing
apply_smooth = False

# Normalize each spectrum to its own maximum before building the heatmap
normalize = True

# Normalize the entire dataset to the global maximum intensity
normalize_all = False

# Plot each individual spectrum
plot_traces = False

# Contour plot settings
plot_title = ''
cmap = 'inferno'

# Fit each spectrum with Lorentzians
do_fit = False

# Fit parameters: [amp1, cen1, wid1, amp2, cen2, wid2, ...]
p0 = [
    20/260, 1.31, 0.006,
    90/260, 1.33, 0.006,
    260/260, 1.34, 0.004,
    80/260, 1.35, 0.005,
    20/260, 1.355, 0.005,
]

bounds = [
    [0, None], [1.307, 1.315], [0.001, 0.1],
    [0, None], [1.3,   1.33],  [0.001, 0.01],
    [0, None], [1.32,  1.35],  [0.001, 0.01],
    [0, None], [1.33,  1.356], [0.001, 0.01],
    [0, None], [1.345, 1.36],  [0.001, 0.01],
]

# ---- End of user settings ---- #

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).parent.parent

from processing_package import (
    read_spectrum, get_b_fields, get_heatmap_data, plot_heatmap,
    diffL, Lorentz, diffquad, quadratic,
)


def load_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    base = config_path.parent
    for key in ('data_dir', 'processed_data_dir', 'reports_dir', 'pl_dir',
                'pl_bfield_dir'):
        if key in cfg:
            cfg[key] = (base / cfg[key]).resolve()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PL vs. B-field processing.")
    parser.add_argument(
        '--project',
        default=None,
        help="Name of a directory in the same parent folder as this script "
             "containing config.yaml",
    )
    args, _ = parser.parse_known_args()

    if args.project:
        config_path = ROOT_DIR.parent / args.project / 'config.yaml'
    else:
        config_path = ROOT_DIR.parent / '07_CrSBr_Cryo4' / 'config.yaml'

    cfg = load_config(config_path)
    directory = cfg['pl_bfield_dir']

    file_paths = glob.glob(os.path.join(directory, '*.csv'))
    bfields, bfield_list = get_b_fields(file_paths)

    yvar_all = []
    intensity_all = []
    maxes = []
    peak_locs = {}
    peak_amps = {}

    for i, (bfield, file_path) in enumerate(bfield_list):
        energy, intensity, max_intensity = read_spectrum(
            file_path, normalize=normalize, xEnergy=True, umBlaze=False
        )
        maxes.append(max_intensity)

        yvar, zvar = get_heatmap_data(
            energy, intensity,
            plot_with_mask=plot_with_mask,
            min_energy=min_energy,
            max_energy=max_energy,
            apply_smooth=apply_smooth,
        )
        yvar_all.append(yvar)
        intensity_all.append(zvar)

        if do_fit:
            fit = minimize(diffL, p0, args=(yvar, zvar), bounds=bounds).get('x')
            n_peaks = len(p0) // 3
            peak_locs[str(bfield)] = [fit[3*j + 1] for j in range(n_peaks)]
            peak_amps[str(bfield)] = [fit[3*j]     for j in range(n_peaks)]

            if plot_traces:
                plt.figure(dpi=400)
                plt.title(f"{bfield} T")
                plt.plot(yvar, zvar)
                plt.plot(yvar, Lorentz(fit, yvar))
                plt.show()
        elif plot_traces:
            plt.figure(dpi=400)
            plt.title(f"{bfield} T")
            plt.plot(yvar, zvar)
            plt.show()

    if normalize_all:
        global_max = max(maxes)
        intensity_all = [z / global_max for z in intensity_all]

    plot_heatmap(bfields, yvar_all, intensity_all, plot_title, cmap=cmap)

    # ---- Peak positions vs. B-field (only when do_fit is True) ---- #
    if do_fit and peak_locs:
        n_peaks = len(p0) // 3
        by_peak = [[] for _ in range(n_peaks)]
        for bfield in bfields:
            locs = peak_locs[str(bfield)]
            for j in range(n_peaks):
                by_peak[j].append(locs[j])

        pquad_init = [-3, 0, 1.4]
        for i in range(n_peaks):
            quad_fit = minimize(
                diffquad, pquad_init, args=(bfields, by_peak[i])
            ).get('x')
            a = round(quad_fit[0], 5)
            b = round(quad_fit[1], 5)
            c = round(quad_fit[2], 2)
            label = f"$y_{i+1}$ = {a}$x^2$ + {b}$x$ + {c}"
            plt.figure(dpi=400)
            plt.plot(bfields, by_peak[i], "r.")
            plt.title(f"Peak #{i+1}, {label}")
            plt.plot(bfields, quadratic(quad_fit, bfields))
            plt.show()
