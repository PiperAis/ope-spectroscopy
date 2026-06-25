# TRR / Steady-State Analysis Package

Python package for processing and analyzing optical spectroscopy data — time-resolved reflectance (TRR) and steady-state photoluminescence (PL), including B-field-dependent measurements. Integrated with an Obsidian lab notebook for automatic note and report generation.

---

## Setup

**Requirements:** Python 3.10+, a virtual environment (recommended).

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

# 3. Install the package in editable mode
pip install -e .
```

Dependencies (numpy, scipy, xarray, matplotlib, pandas, netcdf4, pyyaml) are declared in `pyproject.toml` and installed automatically.

---

## Project layout

This repo is the **package** — it doesn't contain your data or your project-specific scripts. Each experiment campaign lives in its own project folder alongside your data:

```
MyExperiment/              ← your project folder (not in this repo)
├── config.yaml            ← points to your data; all paths live here
├── TRR/
│   ├── raw_data/
│   │   └── 2026-07-01/   ← dated folder of raw .csv files from TimeSpec
│   ├── processed/
│   └── reports/
└── PL/
    ├── raw_data/
    └── processed/
```

The package contains no hardcoded paths. All locations come from `config.yaml`.

---

## config.yaml

Create a `config.yaml` in your project folder. Paths are resolved relative to the config file:

```yaml
data_dir: ./TRR/raw_data
processed_data_dir: ./TRR/processed
reports_dir: ./TRR/reports
pl_dir: ./PL/raw_data
processed_pl_dir: ./PL/processed
pl_bfield_dir: ./PL/raw_data/2026-07-01/bfield-sweep   # folder of B-field PL spectra

# Optional — only needed for Obsidian vault integration
vault_dir: /absolute/path/to/your/obsidian/vault
```

---

## Filename convention

Raw data files must follow this naming format (dash-delimited, `p` as decimal point):

```
##-[sample]-[temp]K-[bfield]T-[scale]SF-[R0]mV-[extra tokens]
```

Example: `01-flake1a-1p6K-0p2T-70kSF-120mV-2pass-offset1`

- `##` — per-experiment dataset ID (two digits)
- Tags are identified by their suffix/format, not by position
- Unrecognized tokens are preserved as a freeform list
- Use `n` prefix for negative fields: `n0p2T` → −0.2 T

---

## Running the scripts

Driver scripts live in `scripts/`. Each takes `--config` pointing to your `config.yaml` and `--date` (or similar) for the data folder to process. Run them from the project folder with the venv active.

### TRR processing (report only)

```bash
python /path/to/scripts/Full_TRR_Report_Generator.py --config config.yaml --date 2026-07-01
```

Runs the full TRR pipeline (noise removal → normalize → subtract decay → FFT) and generates a per-step audit report in `reports_dir`.

### TRR processing + Obsidian vault integration

```bash
python /path/to/scripts/TRR_process_to_vault.py --config config.yaml --date 2026-07-01
```

Same as above, then copies the report folder to the vault and auto-generates one Obsidian dataset note per file, pre-filled with filename metadata.

### PL vs. B-field contour plot

```bash
python /path/to/scripts/PL_bfield_driver.py --config config.yaml
```

Generates a heatmap of PL intensity vs. energy and magnetic field from the folder specified by `pl_bfield_dir` in config.

### Steady-state PL

```bash
python /path/to/scripts/steady_state_basics.py --config config.yaml
```

Loads and plots steady-state PL spectra. Fitting workflow is under active development.

---

## Package modules

| Module | Contents |
|---|---|
| `trr_dataset` | xarray accessor (`trrxr`) for TRR data; loading and dataset-level operations |
| `processing_state` | `ProcessingState` — linear pipeline orchestrator for TRR (noise → normalize → subtract → FFT) |
| `steady_state` | Loading and helper functions for steady-state PL data |
| `fitting_functions` | Shared fitting primitives (exponential decay, Lorentzian, etc.) |
| `fft_functions` | FFT-based noise analysis utilities |
| `noise_remover` | Noise removal step used in the TRR pipeline |
| `utilities` | Filename parsing, config loading, and small shared helpers |

---

## Notes

- The package is installed editable (`pip install -e .`). Any edits to `processing_package/` take effect immediately — no reinstall needed.
- The fitting functionality in `PL_bfield_driver.py` (Lorentzian peak fitting per spectrum) is currently out of date and not reliable.
- If you use Obsidian, set `vault_dir` to the absolute path of your vault root. The integration writes notes under `vault_dir/<date>/`.
