"""
Driver script for interactive PL fitting and cross-dataset comparison.

INTERACTIVE FITTER — run as a plain script (NOT a #%% cell):
    python scripts/steady_state_driver.py --project MyExperiment

    A folder browser opens in the project's PL data directory — pick a sweep
    (date) folder and every spectrum in it is queued, in filename order
    (metadata.txt and other non-spectrum files are skipped). Add --pick-files
    to choose individual files instead, or --files to skip the browser.

    Left-click the spectrum to place peak markers; right-click to remove one.
    E min / E max boxes set the fit window (persists across files; the
    toolbar zoom is view-only). "Baseline" button: constant or linear.
    "Carry peaks to next file" seeds the next spectrum's markers.
    Buttons: Fit Gaussian | Fit Lorentzian | Clear Peaks | Save | ← Back | Next →
    Back (or Next onto a file already visited) restores that file's markers,
    fit, range and baseline; the title shows whether the fit has been saved.

    Save writes, per spectrum:
      <processed_pl_dir>/fit_results.csv          one row per peak, with 1σ errors
      <processed_pl_dir>/plots/<stem>_fit.png     fit overlay
      <processed_pl_dir>/<sweep>/<stem>.nc        raw / baseline / subtracted data
                                                  (+ fit settings and results in attrs)

CONTOUR PLOT — PL intensity vs. energy and B-field:
    python scripts/steady_state_driver.py --project MyExperiment --mode contour-plot

    A directory browser opens in the project's processed PL directory — pick
    a sweep folder of fitted .nc spectra. A raw CSV sweep folder also works
    (raw data only). Pass --files to skip the browser (a single directory,
    or an explicit list of spectrum paths).

    Live controls: colormap, E min / E max (eV), normalize to global max, and
    — when every spectrum in the folder has been fitted — a "Data" button to
    switch between baseline-subtracted (default, opens on the fit window) and
    raw. Save writes <processed_pl_dir>/<sample>_bfield_contour[_subtracted].png.

COMPARISON PLOTS — run the #%% cells below interactively in VS Code
    after fitting to build up fit_results.csv.
"""

#%%
# ---- Imports and config (uncomment and run this cell first for interactive use) ---- #

from opespec.steady_state import PLFitter, BfieldContourBuilder, load_fit_results, plot_peak_param
from opespec.utilities import load_config, select_files, select_directory

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
                        help="Explicit spectrum paths, or a single directory — skips the browser")
    parser.add_argument('--pick-files', action='store_true',
                        help="Fit mode: pick individual files instead of a whole folder")
    parser.add_argument('--mode', choices=['fit', 'contour-plot'], default='fit',
                        help='What type of processing to execute. Options: fit, contour-plot.')
    args = parser.parse_args()

    cfg = load_config(args.project)

    if args.mode == 'contour-plot':
        if args.files:
            files = [Path(f) for f in args.files]
        else:
            files = select_directory(Path(cfg['processed_pl_dir']))
            if not files:
                raise SystemExit("No directory selected.")
        print("Opening contour-plot builder...")
        builder = BfieldContourBuilder(files, output_dir=Path(cfg['processed_pl_dir']))
        builder.run()

    if args.mode == 'fit':
        if args.files:
            files = [Path(f) for f in args.files]
        elif args.pick_files:
            files = select_files(Path(cfg['pl_dir']))
            if not files:
                raise SystemExit("No files selected.")
        else:
            files = select_directory(Path(cfg['pl_dir']))
            if not files:
                raise SystemExit("No directory selected.")
        fitter = PLFitter(files, output_dir=Path(cfg['processed_pl_dir']))
        print(f"Opening fitter for {len(fitter.files)} file(s)...")
        fitter.run()
    

