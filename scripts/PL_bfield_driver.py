"""
Driver script for PL vs. B-field processing.

Generates a contour (heatmap) plot of PL intensity as a function of
energy and magnetic field. Optionally fits each spectrum with Lorentzians
and plots peak positions vs. field, however this functionality is out of
date and needs to be updated to work with the current version of the
package.

Data directory is read from config.yaml (pl_bfield_dir key).
Pass --project <folder name> to use a different project's config;
defaults to 07_CrSBr_Cryo4.

Usage:
    python PL_bfield_driver.py
    python PL_bfield_driver.py --project <project_folder_name>
"""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from opespec import bfield_contour_plot
from opespec.utilities import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PL vs. B-field processing.")
    parser.add_argument('--project', required=True,
                        help="Project folder name")
    args = parser.parse_args()

    cfg = load_config(args.project)

    # experiment_date = '2026-03-03'
    data_path = Path(cfg['pl_bfield_dir'])

    bfield_contour_plot(data_path)

# ---- Old code, to repurpose for fitting and tracking peaks vs bfield ---- #

# Fit parameters: [amp1, cen1, wid1, amp2, cen2, wid2, ...]
# p0 = [
#     20/260, 1.31, 0.006,
#     90/260, 1.33, 0.006,
#     260/260, 1.34, 0.004,
#     80/260, 1.35, 0.005,
#     20/260, 1.355, 0.005,
# ]

# bounds = [
#     [0, None], [1.307, 1.315], [0.001, 0.1],
#     [0, None], [1.3,   1.33],  [0.001, 0.01],
#     [0, None], [1.32,  1.35],  [0.001, 0.01],
#     [0, None], [1.33,  1.356], [0.001, 0.01],
#     [0, None], [1.345, 1.36],  [0.001, 0.01],
# ]

# # ---- Peak positions vs. B-field (only when do_fit is True) ---- #
# if do_fit and peak_locs:
#     n_peaks = len(p0) // 3
#     by_peak = [[] for _ in range(n_peaks)]
#     for bfield in bfields:
#         locs = peak_locs[str(bfield)]
#         for j in range(n_peaks):
#             by_peak[j].append(locs[j])

#     pquad_init = [-3, 0, 1.4]
#     for i in range(n_peaks):
#         quad_fit = minimize(
#             diffquad, pquad_init, args=(bfields, by_peak[i])
#         ).get('x')
#         a = round(quad_fit[0], 5)
#         b = round(quad_fit[1], 5)
#         c = round(quad_fit[2], 2)
#         label = f"$y_{i+1}$ = {a}$x^2$ + {b}$x$ + {c}"
#         plt.figure(dpi=400)
#         plt.plot(bfields, by_peak[i], "r.")
#         plt.title(f"Peak #{i+1}, {label}")
#         plt.plot(bfields, quadratic(quad_fit, bfields))
#         plt.show()
