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
      <processed_pl_dir>/<sweep>/<stem>.nc        raw data + every fit session's baseline /
                                                  subtracted data, settings and results
                                                  (re-save replaces this session's fit)
      <processed_pl_dir>/fit_results.csv          regenerated from the .nc files: one row per
                                                  peak per session, with 1σ errors, fit window,
                                                  RMSE/R², saved_at; tagged experiment + fit_number
      <processed_pl_dir>/<sweep>/plots/fits_<n>/<stem>_fit.png
                                                  fit overlay; <n> counts up per session

EDIT A FIT SESSION — reopen session n and save into it instead of a new one:
    python scripts/steady_state_driver.py --project MyExperiment --edit-session 3
    python scripts/steady_state_driver.py --project MyExperiment --edit-session

    Pick the sweep folder as usual. Without a number, the folder's sessions
    are listed (spectra, last saved) and you're asked for one. Spectra fitted
    in that session open with the saved fit drawn as stored (no refit);
    others open blank and can be added. Save replaces the spectrum's fit in
    that session and overwrites its fits_<n>/ plot.

    --sync-obsidian: when the fitter window closes, every fitted spectrum in the
    sweep folder(s) gets a note <vault_dir>/<Project>/<sweep> PL/Dataset <N>.md
    (project dashes → spaces, e.g. HS-2 → HS 2). Properties: sample/spot/field/
    temperature plus any other metadata.txt keys (filename wins, as at load), and
    its latest fit plot copied to plots/ and embedded. Re-syncing rewrites only
    those generated properties; your Notes and body are kept.

SYNC ONLY — Obsidian notes from fits already saved (e.g. forgot --sync-obsidian):
    python scripts/steady_state_driver.py --project MyExperiment --mode sync
    python scripts/steady_state_driver.py --project MyExperiment --mode sync --files PL/processed/2026-11-03

    Pick a processed sweep folder (or pass it with --files); no window opens.
    Each spectrum's note uses its most recent saved fit.

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

from opespec.steady_state import PLFitter, load_fit_results, plot_peak_param
from opespec.pl_sweep import SweepContourBuilder, brightest_point
from opespec.pl_fit_compare import FitCompareViewer, track_branches, track_peak, fit_branch
from opespec.utilities import load_config, select_files, select_directory
from opespec.obsidian_integration import sync_pl_sweep

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

# Track the peak brightest at B = 0 through each field sweep (the brightest branch below):
# plot_peak_param(track_peak(results), x_col='field', y_col='energy', group_by='experiment',
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
def ask_edit_session(cfg: dict, files) -> int:
    """List the fit sessions of the chosen sweep folder(s) and ask which to edit."""
    from pathlib import Path
    paths = [Path(f) for f in (files if isinstance(files, list) else [files])]
    experiments = sorted({p.name if p.is_dir() else p.parent.name for p in paths})
    csv_path = Path(cfg['processed_pl_dir']) / 'fit_results.csv'
    if not csv_path.exists():
        raise SystemExit(f"No fit results yet: {csv_path}")
    df = load_fit_results(csv_path)
    df = df[df['experiment'].astype(str).isin(experiments)]
    if df.empty:
        raise SystemExit(f"No saved fit sessions for {', '.join(experiments)}.")
    for experiment, rows in df.groupby(df['experiment'].astype(str)):
        print(f"Fit sessions in {experiment}:")
        for n, session in rows.groupby('fit_number'):
            saved = session['saved_at'].dropna()
            last = saved.max() if len(saved) else 'unknown'
            print(f"  {int(n):>3}: {session['filename'].nunique()} spectra, last saved {last}")
    try:
        return int(input("Session to edit: "))
    except ValueError:
        raise SystemExit("Not a session number.")


def sync_obsidian(cfg: dict, project: str, sweeps) -> None:
    """Write/update vault dataset notes for each named sweep folder."""
    from pathlib import Path
    if 'vault_root' not in cfg:
        raise SystemExit("Obsidian sync needs vault_dir in machine_settings.yaml.")
    # Vault project folders use spaces where project folders use dashes (HS-2 → HS 2).
    vault_project = cfg['vault_root'] / project.replace('-', ' ')
    for sweep in sorted(sweeps):
        sweep_dir = Path(cfg['processed_pl_dir']) / sweep
        if not sweep_dir.is_dir():
            print(f"Nothing saved for {sweep}; skipping Obsidian sync.")
            continue
        expt_dir = sync_pl_sweep(sweep_dir, Path(cfg['pl_dir']) / sweep, vault_project)
        print(f"Obsidian notes → {expt_dir}")


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
    parser.add_argument('--mode', choices=['fit', 'contour-plot', 'compare', 'sync'], default='fit',
                        help='What type of processing to execute. Options: fit, contour-plot, '
                             'compare, sync (Obsidian notes only, from saved fits).')
    parser.add_argument('--sweep', choices=['field', 'temperature'], default='field',
                        help="Contour-plot mode: the variable stepped across the sweep folder.")
    parser.add_argument('--edit-session', nargs='?', type=int, const=-1, default=None,
                        metavar='N',
                        help="Fit mode: save into existing fit session N instead of a new one "
                             "(its saved fits are restored). Without N, list sessions and ask.")
    parser.add_argument('--sync-obsidian', action='store_true',
                        help="Fit mode: after the window closes, write/update Obsidian dataset "
                             "notes for the fitted sweep folder(s).")
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
        edit = args.edit_session
        if edit == -1:
            edit = ask_edit_session(cfg, files)
        fitter = PLFitter(files, output_dir=Path(cfg['processed_pl_dir']), edit_session=edit)
        print(f"Opening fitter for {len(fitter.files)} file(s)...")
        fitter.run()

        if args.sync_obsidian:
            sync_obsidian(cfg, args.project, {f.parent.name for f in fitter.files})

    if args.mode == 'sync':
        if args.files:
            sweeps = {Path(f).name for f in args.files}
        else:
            chosen = select_directory(Path(cfg['processed_pl_dir']))
            if not chosen:
                raise SystemExit("No directory selected.")
            sweeps = {chosen.name}
        sync_obsidian(cfg, args.project, sweeps)


