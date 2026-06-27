"""
Driver script for interactive PL fitting and cross-dataset comparison.

INTERACTIVE FITTER — run as a plain script (NOT a #%% cell):
    python scripts/steady_state_basics.py --project 07_CrSBr_Cryo4

    A file browser opens in the project's PL data directory — navigate to a
    date subfolder and select one or more CSV files (Ctrl+A for all).

    Left-click the spectrum to place peak markers.
    Right-click a marker to remove it.
    Zoom with matplotlib's zoom tool to restrict the fit range.
    Buttons: Fit Gaussian | Fit Lorentzian | Clear Peaks | Save | Next →

    Results append to <processed_pl_dir>/fit_results.csv (one row per peak).
    Fit-overlay plots save to <processed_pl_dir>/plots/<stem>_fit.png.

COMPARISON PLOTS — run the #%% cells below interactively in VS Code
    after fitting to build up fit_results.csv.
"""

#%%
# ---- Imports and config (uncomment and run this cell first for interactive use) ---- #

from opespec.steady_state import PLFitter, load_fit_results, plot_peak_param
from opespec.utilities import load_config, select_files

## Set project name here for interactive cell use:
# project_name = 'CrSBr-4'
# cfg = load_config(project_name)

#%%
# ---- Comparison plots ---- #
# Run these cells interactively after fitting to explore results.
# Adjust x_col, y_col, peak_index, and group_by as needed.

# results = load_fit_results(Path(cfg['processed_pl_dir']) / 'fit_results.csv')
# print(results.to_string())

# Peak energy vs magnetic field, one series per sample:
# plot_peak_param(results, x_col='field', y_col='energy', peak_index=0, group_by='sample')

# Peak width vs temperature:
# plot_peak_param(results, x_col='temperature', y_col='width', peak_index=0)

# All peaks on one plot, coloured by peak index:
# plot_peak_param(results, x_col='field', y_col='energy', group_by='peak_index')

#%%
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Interactive PL fitter.")
    parser.add_argument('--project', required=True,
                        help="Project folder name.")
    parser.add_argument('--files', nargs='+', default=None,
                        help="Explicit CSV paths — skips the file browser")
    args = parser.parse_args()

    cfg = load_config(args.project)

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = select_files(Path(cfg['pl_dir']))
        if not files:
            raise SystemExit("No files selected.")

    print(f"Opening fitter for {len(files)} file(s)...")
    fitter = PLFitter(files, output_dir=Path(cfg['processed_pl_dir']))
    fitter.run()
