# OPE Spectroscopy Package

Python package for processing and analyzing optical spectroscopy data — steady-state photoluminescence (PL), including B-field-dependent measurements, and time-resolved reflectance (TRR) ultrafast measurements. Integrated with an Obsidian lab notebook for automatic note generation (optional).

---

## Setup

**Requirements:** Python 3.10+, a virtual environment (recommended).

### Where to install

At time of writing, our group stores data on Box. This package is version-controlled using git, which conflicts with Box's sync behavior, so you should install this package on your *local* drive (for example under C:/Users/\<yourname\>/Code) and use git to pull updates (and to push them, if you choose to contribute to development!)

To do this, navigate to the folder you'd like to install in (either in a terminal or using your file explorer, then right click the folder and select "Open in Terminal") then execute the commands below.


1. Clone the repo and navigate into the directory
```bash
git clone https://github.com/PiperAis/ope-spectroscopy
cd ope-spectroscopy
```

2. Create and activate a virtual environment inside the ope-spectroscopy folder [recommended - this creates a separate environment to install this package and its dependencies, which prevents potential conflicts with other packages you may use elsewhere.]
```bash
python -m venv .venv

# Windows (cmd)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

3. Install the package
```bash
pip install .

# If you plan to make edits, use editable mode
pip install -e .
```

Dependencies (numpy, scipy, xarray, matplotlib, pandas, netcdf4, pyyaml) are declared in `pyproject.toml` and installed automatically.

---

## Project layout

This repo is the **package** — it doesn't contain your data or your project-specific scripts. It uses 'machine_settings.yaml' and 'config.yaml' to locate your data and other relevant files.

The package assumes the following structure in your data storage folder by default (if you use the default config contained in 'config.example.yaml'). You can change `config.yaml` if you like to use a different default structure.

```
<your name> Data              ← your data folder on Box (or wherever you keep your data)
└──MyExperiment/              ← project folder
    ├── PL/
    │   ├── raw_data/
    │   │   └── 2026-06-25/   ← dated folder of raw .csv files
    │   └── processed/
    ├── TRR/                  ← folder for transient reflectance, if you use that technique
    │   ├── raw_data/
    │   │   └── 2026-07-01/    ← dated folder of raw files from TimeSpec
    │   ├── processed/
    │   └── reports/
    └── other-measurement-type/
        ├── raw_data/
        └── processed/
```

---

## machine_settings.yaml and config.yaml

Create a `config.yaml` and a `machine_settings.yaml` in your project folder by copying and renaming the example files and editing to match your own project structure if needed. You can get paths to the relevant folders for machine_settings by right-clicking in file explorer and selecting `Copy as Path`.

`machine_settings.yaml` - tells the package where your highest-level data folder is, that all of your project folders live inside of.
```yaml
projects_dir: C:/path/to/your/projects
# Optional: only needed if you want to use integration with an Obsidian lab notebook.
vault_root: C:/Vaults/your-vault 
```

`config.yaml` - specifies where (relative to any given project folder) each type of data can be found. You can add directories for additional types of measurements as needed, or modify this to match whatever structure you use, just keep the keys (labels for the directory) the same.
```yaml
trr_dir: ./TRR/raw_data
processed_trr_dir: ./TRR/processed
reports_dir: ./TRR/reports
pl_dir: ./PL/raw_data
processed_pl_dir: ./PL/processed
```

---

## Filename convention

Raw data files must follow this naming format using dash-delimited "tokens" for each experimental parameter:

```
##-[sample]-[temp]K-[bfield]T-[scale]SF-[R0]mV-[extra tokens]
```

Example: `01-flake1a-1p6K-0p2T-70kSF-120mV-2pass-offset1`

- `##` — per-experiment dataset ID (two digits)
- Tags are identified by their suffix/format, not by position, so the order is not important.
- Use `p` for decimals
- Use `n` prefix for negative fields: `n0p2T` → −0.2 T
- Unrecognized tokens are preserved as a freeform list in the Obsidian lab notebook if used.

### Folder metadata (`metadata.txt`)

If a parameter is the same for every file in a folder (e.g. the sample), you can leave it out of the filenames and put it in a `metadata.txt` in that folder instead, one `key: value` per line:

```
# Constant for every file in this folder
sample: Flake 1a
temperature: 1.6
scale_factor: 70000
```

- Recognized keys: `sample`, `spot`, `field` (T), `temperature` (K), `scale_factor`, `rzero` (mV). Any other keys are kept too.
- Write values in their final form, not filename form: `Flake 1a` not `flake1a`, `1.6` not `1p6K`, `-0.2` not `n0p2T`, `70000` not `70kSF`.
- A value in the filename overrides `metadata.txt`, with a warning if the two disagree.
- `#` starts a comment.

---

## Running the scripts

Driver scripts (applications that use the package to process data) live in `scripts/`. Each takes your project name as an input. You can run them through the terminal or in an interactive environment. The `project_name` variable (if running interactively) or `--project` argument (if running in a terminal) should be a string that matches the name of your project folder.

### TRR processing (report only)

```bash
python /path/to/scripts/trr_driver.py --project CrSBr-3
```

Runs the full TRR pipeline (noise removal → normalize → subtract decay → FFT) and generates a per-step audit report in `reports_dir`.

### TRR processing + Obsidian vault integration

```bash
python /path/to/scripts/trr_driver.py --project CrSBr-3 --sync-obsidian
```

### Steady-state PL — interactive fitter

```bash
python /path/to/scripts/steady_state_driver.py --project MyExperiment
```

Opens a folder browser in the project's `pl_dir`. Pick a sweep (date) folder and every spectrum in it is queued in filename order; `metadata.txt` and other non-spectrum files are skipped. Pass `--pick-files` to choose individual files instead, or `--files` to skip the browser.

An interactive matplotlib window opens:

- Left-click to place peak markers; right-click to remove one.
- **Shift+left-click** a marker to put it in or take it out of the shared-width group (magenta markers). Peaks in the group are fitted with one common FWHM, which stops overlapping peaks trading width and keeps line shapes consistent across a sweep. After a fit, the **Shared FWHM (meV)** box shows the fitted width, or you can type a value. With **Lock shared width** ticked, fits use the box value instead of fitting it, and a locked width is reported with `width_err` = 0. The lock and the value carry over to the next file. A typical sweep: fit B = 0 with the width unlocked, tick Lock, then carry the peaks through the rest of the sweep.
- **E min / E max** boxes set the fit window. It carries over to the next file. Toolbar zoom only changes the view, not the fit.
- **Baseline** cycles constant ↔ linear. The baseline is fit together with the peaks, and the view shows the baseline-subtracted data.
- **Ctrl+left-click** a marker to put it in or take it out of the equal-spacing group (thick dash-dot markers). Peaks in this group, sorted by energy, are fitted as an evenly spaced ladder E₀, E₀+Δ, E₀+2Δ, …, so only E₀ and one spacing Δ are fitted for their centres. Δ ± error prints with the fit and is saved. The group needs at least two peaks. E₀ is held within ±8 meV of the lowest clicked marker, and Δ within ±8 meV of the average clicked spacing. A peak can be in both groups.
- **Carry peaks to next file** seeds the next spectrum's markers from this one's fitted peaks, including their width and spacing groups.
- **Fit Gaussian** / **Fit Lorentzian** run the fit. Results with 1σ errors, RMSE and R² print to the console.
- The title shows whether the current fit has been saved.

**Save** writes:

- the fit into the processed spectrum `processed_pl_dir/<experiment>/<stem>.nc`. The file holds `raw` and, for every fit session of that spectrum (along a `fit_number` dimension), `baseline` and `subtracted` plus the fit settings, per-peak results and a `saved_at` time. Readers (`load_pl_data`, the contour builder, Obsidian sync) use each spectrum's latest session unless given `fit_number=`.
- `processed_pl_dir/fit_results.csv`, regenerated from the `.nc` files on every save: a row per peak per session, with 1σ errors, `width_shared` / `width_locked` / `energy_spaced` flags (plus `spacing` / `spacing_err` on spaced peaks), the fit window (`fit_emin`/`fit_emax`), fit quality (`rmse`, `r_squared`) and `saved_at`, tagged with `experiment` (the sweep folder) and `fit_number`. Edit the `.nc` files, not the CSV; the CSV is overwritten.
- a clean fit-overlay PNG to `processed_pl_dir/<experiment>/plots/fits_<n>/<stem>_fit.png`.

Each fitter run is a new fit session, and `n` (= `fit_number`) counts up from the highest existing `fits_<n>` folder for that experiment. Re-saving a spectrum in the same session replaces that session's fit and plot. Earlier sessions are kept, so iterations can be compared. `track_branches` and `track_peak` use each spectrum's latest fit by default; pass `fit_number=` to use a specific session. `fit_number` 0 holds fits saved before sessions were numbered.

**Editing a fit session:** `--edit-session N` (or `PLFitter(..., edit_session=N)`) saves into existing session N instead of starting a new one. Pass `--edit-session` with no number to list the folder's sessions and pick one. Spectra fitted in that session open with the saved fit drawn exactly as stored (no refit), along with their markers, fit window and baseline type. If the fit had a shared-width group, its shared FWHM and Lock state are restored too, so a refit keeps the same constraint. Spectra not in the session open as usual and can be added to it. Save replaces the spectrum's fit in that session and overwrites its `fits_<N>/` plot. Sessions migrated from the CSV alone open with markers only and need a refit before saving.

**Converting older processed folders:** `.nc` files saved before the session layout hold only their last fit. Convert a project once with `python scripts/migrate_pl_sessions.py --project <name>` (a dry run that checks nothing changes), then again with `--write`. Sessions that were only in `fit_results.csv` come over without a baseline, so they can't be shown as baseline-subtracted until refit. Originals are kept in `processed_pl_dir/pre_migration_backup/` and `fit_results_pre_migration.csv`.

**Obsidian sync:** add `--sync-obsidian` to fit mode. When the window closes, every fitted spectrum in the sweep folder gets a note `<vault_dir>/<Project>/<sweep> PL/Dataset <N>.md`. The project name has its dashes replaced by spaces (`HS-2` → `HS 2`), and that folder must already exist. Metadata is resolved as at load time: the filename, then the raw folder's `metadata.txt`. Each note gets Sample / Spot / Bfield / Temperature plus any extra `metadata.txt` keys, and the spectrum's latest fit plot, which is copied to `plots/` and embedded. If you fitted without the flag, `--mode sync` (with a folder browser, or `--files <processed sweep folder>`) syncs from the saved fits without opening the fitter. Re-syncing rewrites only those generated properties and keeps your Notes, other properties and the note body. Spectra whose files share a `##` prefix are named by the full stem instead.

**Next →** advances to the next file and closes the window after the last one. **← Back** (or Next onto a file you've already visited) restores that file's markers, fit, E range and baseline, so you can save a forgotten fit or adjust and refit it. Run the fitter as a plain script, not in a `#%%` cell; the Jupyter kernel intercepts matplotlib clicks.

### PL contour plot (vs. B-field or temperature)

```bash
python /path/to/scripts/steady_state_driver.py --project CrSBr-3 --mode contour-plot
python /path/to/scripts/steady_state_driver.py --project CrSBr-3 --mode contour-plot --sweep temperature
```

Opens a directory browser in the project's `processed_pl_dir`. Pick a sweep folder of fitted `.nc` spectra. A raw CSV sweep folder also works, but shows raw data only. Pass `--files` to skip the browser, with either a single directory or an explicit list of spectrum paths.

An interactive matplotlib window opens with a heatmap of PL intensity vs. energy and the sweep variable: magnetic field by default, or temperature with `--sweep temperature`. Each spectrum's value comes from its filename or the folder's `metadata.txt`. Live controls:

- colormap name
- lower/upper energy bound (eV)
- a **Normalize** button that cycles Off → Global max (all spectra against the dataset's max) → Per spectrum (each spectrum against its own max within the energy window)
- optional Savitzky–Golay smoothing along energy (checkbox + window in points). This is display only: the data isn't changed.

When every spectrum in the folder has been fitted, the plot opens on baseline-subtracted data over the shared fit window, and a **Data** button switches to raw. **Save** writes the current figure to `processed_pl_dir/<sample>_<tag>_contour[_subtracted][_sg<N>].png`, where `<tag>` is `bfield` or `temperature`.

### PL fit comparison (tracked peaks vs. B-field)

```bash
python /path/to/scripts/steady_state_driver.py --project CrSBr-3 --mode compare
```

Reads `processed_pl_dir/fit_results.csv`. In each experiment (sweep folder), every peak of the B ≈ 0 fit starts a branch, which is followed field to field by nearest energy (`track_branches`). The window plots peak energy vs field: one colour per experiment, one marker per branch, `energy_err` error bars, and all fitted peaks in grey behind them for context.

- Hollow markers are fitted peaks claimed by more than one branch (peaks merged at high field).
- A black × flags a doubtful fit: R² below `R2_MIN`, or `energy_err`/`width` above `REL_ERR_MAX` (class constants on `FitCompareViewer`).
- The dashed lines are a weighted quadratic E0 + a·B + b·B² per branch (`fit_branch`). Its coefficients ± errors are listed under the plot. Shared points are left out of this fit unless **Include shared points** is ticked.
- Checkboxes show or hide each branch. **Session** cycles between each spectrum's latest fit and a specific `fit_number`.
- **Save** writes `processed_pl_dir/branch_compare_<session>.png` and `branch_fits_<session>.csv` (coefficients of the visible branches, in eV, eV/T and eV/T²).

Run it as a plain script, like the fitter.

After fitting, open `steady_state_driver.py` in VS Code (or your Python environment of choice) and run the `#%%` comparison-plot cells interactively to plot fit parameters across datasets (e.g. peak energy vs B-field, width vs temperature).

---

## Package modules

| Module | Contents |
|---|---|
| `trr_dataset` | xarray accessor (`trrxr`) for TRR data; loading and dataset-level operations |
| `processing_state` | `ProcessingState` — linear pipeline orchestrator for TRR (noise → normalize → subtract → FFT) |
| `steady_state` | `load_pl_data`; `PLFitter` interactive fitter; `load_fit_results` / `plot_peak_param` for cross-dataset comparison |
| `pl_sweep` | Spectra vs. a sweep variable (`SWEEPS`: field or temperature): `SweepContourBuilder` interactive contour-plot builder; `brightest_point` |
| `pl_fit_compare` | Cross-dataset comparison of saved fits: `track_branches` (all B = 0 peaks through a field sweep), `track_peak` (the brightest branch only), `fit_branch` (quadratic E(B) trend), `FitCompareViewer` interactive window |
| `fitting_functions` | Shared fitting primitives (exponential decay, Lorentzian, Gaussian, peak finding) |
| `fft_functions` | FFT-based noise analysis utilities |
| `noise_remover` | Noise removal step used in the TRR pipeline |
| `utilities` | `load_config`; filename parsers (`get_field`, `get_temperature`, `get_sample_name`, …); `get_metadata` (filename + folder `metadata.txt`); file helpers |

---



