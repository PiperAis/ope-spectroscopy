"""
Driver script for interactive PL fitting and cross-dataset comparison.

INTERACTIVE FITTER — run as a plain script (NOT a #%% cell):
    python scripts/steady_state_driver.py --project MyExperiment

    A folder browser opens in the project's PL data directory — pick a sweep
    (date) folder and every spectrum in it is queued, in filename order
    (metadata.txt and other non-spectrum files are skipped). Add --pick-files
    to choose individual files instead, or --files to skip the browser.

    Left-click the spectrum to place peak markers; right-click to remove one.
    Shift+left-click a marker to toggle it into the shared-width group (magenta):
    those peaks share one fitted FWHM, shown in the "Shared FWHM (meV)" box.
    "Lock shared width" fits with the box value instead (typed, or the last
    fit's) — e.g. fit B = 0 unlocked, then lock for the rest of the sweep.
    Ctrl+left-click a marker to toggle it into the equal-spacing group (thick
    dash-dot): those peaks are fitted as an even ladder E0, E0+Δ, E0+2Δ, …
    with one fitted spacing Δ. A peak can be in both groups.
    E min / E max boxes set the fit window (persists across files; the
    toolbar zoom is view-only). "Baseline" button: constant or linear.
    "Carry peaks to next file" seeds the next spectrum's markers.
    Buttons: Fit Gaussian | Fit Lorentzian | Clear Peaks | Save | ← Back | Next →
    Back (or Next onto a file already visited) restores that file's markers,
    fit, range and baseline; the title shows whether the fit has been saved.

    Save writes, per spectrum:
      <processed_pl_dir>/fit_results.csv          one row per peak, with 1σ errors, fit
                                                  window, RMSE/R²; tagged experiment +
                                                  fit_number (re-save replaces this session's rows)
      <processed_pl_dir>/<sweep>/plots/fits_<n>/<stem>_fit.png
                                                  fit overlay; <n> counts up per session
      <processed_pl_dir>/<sweep>/<stem>.nc        raw / baseline / subtracted data
                                                  (+ fit settings and results in attrs)

CONTOUR PLOT — PL intensity vs. energy and B-field (or temperature):
    python scripts/steady_state_driver.py --project MyExperiment --mode contour-plot
    python scripts/steady_state_driver.py --project MyExperiment --mode contour-plot --sweep temperature

    A directory browser opens in the project's processed PL directory — pick
    a sweep folder of fitted .nc spectra. A raw CSV sweep folder also works
    (raw data only). Pass --files to skip the browser (a single directory,
    or an explicit list of spectrum paths).

    Live controls: colormap, E min / E max (eV), a "Normalize" button cycling
    Off → Global max → Per spectrum (each against its own max in the E window),
    Savitzky–Golay smoothing along energy (checkbox + window in points;
    display only — saved data is untouched; saved PNG gets an _sg<N> suffix), and
    — when every spectrum in the folder has been fitted — a "Data" button to
    switch between baseline-subtracted (default, opens on the fit window) and
    raw. Save writes <processed_pl_dir>/<sample>_<tag>_contour[_subtracted].png
    (<tag> = bfield or temperature).

COMPARE — tracked peak energies vs B-field from fit_results.csv:
    python scripts/steady_state_driver.py --project MyExperiment --mode compare

    Every peak of each sweep's B ≈ 0 fit starts a branch, followed field to
    field by nearest energy; one colour per experiment (sweep folder).
    Hollow markers: shared by two branches (merged peaks; left out of the
    trend fit unless "Include shared" is ticked). Black ×: doubtful fit
    (low R² or energy error large vs width). Dashed: quadratic trend
    E0 + a·B + b·B², coefficients listed under the plot.
    Controls: branch checkboxes, "Session" (latest or a fit_number), Save →
    <processed_pl_dir>/branch_compare_<session>.png + branch_fits_<session>.csv.

COMPARISON PLOTS — run the #%% cells below interactively in VS Code
    after fitting to build up fit_results.csv.
"""

#%%
# ---- Imports and config (uncomment and run this cell first for interactive use) ---- #

from opespec.steady_state import PLFitter, load_fit_results, plot_peak_param, track_peak
from opespec.pl_sweep import SweepContourBuilder, brightest_point
from opespec.pl_fit_compare import FitCompareViewer, track_branches, fit_branch
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

# Track the peak brightest at B = 0 through the field sweep (nearest energy, field to field):
# plot_peak_param(track_peak(results), x_col='field', y_col='energy', group_by='sample',
#                 title='Tracked peak energy vs field')

# Every B = 0 peak as a branch, and a quadratic E(B) trend per branch (the --mode compare window, as tables):
# branches = track_branches(results)
# for (exp, br), rows in branches.groupby(['experiment', 'branch']):
#     print(exp, br, fit_branch(rows))

# Energy of the highest-intensity data point (no fit) vs field, one sweep folder per sample:
# sweeps = [Path(cfg['processed_pl_dir']) / d for d in ['2026-10-27', '2026-11-03']]
# plot_peak_param(brightest_point(sweeps), x_col='field', y_col='energy', group_by='sample',
#                 title='Brightest-point energy vs field')
# For a temperature sweep: brightest_point(sweeps, sweep='temperature'), then x_col='temperature'.

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
    parser.add_argument('--mode', choices=['fit', 'contour-plot', 'compare'], default='fit',
                        help='What type of processing to execute. Options: fit, contour-plot, compare.')
    parser.add_argument('--sweep', choices=['field', 'temperature'], default='field',
                        help="Contour-plot mode: the variable stepped across the sweep folder.")
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
        builder = SweepContourBuilder(files, output_dir=Path(cfg['processed_pl_dir']),
                                      sweep=args.sweep)
        builder.run()

    if args.mode == 'compare':
        csv_path = Path(cfg['processed_pl_dir']) / 'fit_results.csv'
        if not csv_path.exists():
            raise SystemExit(f"No fit results yet: {csv_path}")
        print("Opening fit comparison...")
        FitCompareViewer(load_fit_results(csv_path), output_dir=Path(cfg['processed_pl_dir'])).run()

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
    

