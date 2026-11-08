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
- **E min / E max** boxes set the fit window. It carries over to the next file. Toolbar zoom only changes the view, not the fit.
- **Baseline** cycles constant ↔ linear. The baseline is fit together with the peaks, and the view shows the baseline-subtracted data.
- **Carry peaks to next file** seeds the next spectrum's markers from this one's fitted peaks.
- **Fit Gaussian** / **Fit Lorentzian** run the fit. Results with 1σ errors, RMSE and R² print to the console.
- The title shows whether the current fit has been saved.

**Save** writes:

- a row per peak to `processed_pl_dir/fit_results.csv`. Each row has 1σ errors, the fit window (`fit_emin`/`fit_emax`) and fit quality (`rmse`, `r_squared`), and is tagged with `experiment` (the sweep folder) and `fit_number`.
- a clean fit-overlay PNG to `processed_pl_dir/<experiment>/plots/fits_<n>/<stem>_fit.png`.
- the processed spectrum to `processed_pl_dir/<experiment>/<stem>.nc`, which is overwritten by the latest save and records its `fit_number`.

Each fitter run is a new fit session, and `n` (= `fit_number`) counts up from the highest existing `fits_<n>` folder for that experiment. Re-saving a spectrum in the same session replaces its CSV rows and plot. Earlier sessions are kept, so iterations can be compared. `track_peak` uses each spectrum's latest fit by default; pass `fit_number=` to use a specific session.

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

After fitting, open `steady_state_driver.py` in VS Code (or your Python environment of choice) and run the `#%%` comparison-plot cells interactively to plot fit parameters across datasets (e.g. peak energy vs B-field, width vs temperature).

---

## Package modules

| Module | Contents |
|---|---|
| `trr_dataset` | xarray accessor (`trrxr`) for TRR data; loading and dataset-level operations |
| `processing_state` | `ProcessingState` — linear pipeline orchestrator for TRR (noise → normalize → subtract → FFT) |
| `steady_state` | `load_pl_data`; `PLFitter` interactive fitter; `load_fit_results` / `plot_peak_param` / `track_peak` for cross-dataset comparison |
| `pl_sweep` | Spectra vs. a sweep variable (`SWEEPS`: field or temperature): `SweepContourBuilder` interactive contour-plot builder; `brightest_point` |
| `fitting_functions` | Shared fitting primitives (exponential decay, Lorentzian, Gaussian, peak finding) |
| `fft_functions` | FFT-based noise analysis utilities |
| `noise_remover` | Noise removal step used in the TRR pipeline |
| `utilities` | `load_config`; filename parsers (`get_field`, `get_temperature`, `get_sample_name`, …); `get_metadata` (filename + folder `metadata.txt`); file helpers |

---



