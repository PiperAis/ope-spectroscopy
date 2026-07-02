"""
Steady-state PL processing: loading, interactive fitting, and comparison.

Key components:
- load_pl_data: load a single PL CSV into an xr.DataArray (energy coords)
- PLFitter: interactive matplotlib fitter — click to place peaks, choose
  Gaussian or Lorentzian lineshape, save results to CSV + PNG
- load_fit_results / plot_peak_param: load fit_results.csv and plot any
  parameter pair across datasets (energy vs field, width vs temperature, etc.)
- bfield_contour_plot: heatmap of PL intensity vs energy and B-field
- process_pl_directory_gaussian/lorentzian: batch fitting helpers (older API)
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.optimize import minimize
from scipy.signal import savgol_filter as smooth
from matplotlib.widgets import Button
from . import fitting_functions as ff
from . import utilities

xr.set_options(keep_attrs=True)

def load_pl_data(filepath: Path) -> xr.DataArray:
    """
    Loads PL CSV data into xarray DataArray.
    Assumes columns: Wavelength, Intensity.
    Converts wavelength to energy in eV.
    """
    data = pd.read_csv(filepath, delimiter=',')
    wavelength = data['Wavelength'].values
    wavelength = wavelength
    energy = 1239.8 / wavelength # type: ignore
    intensity = data['Intensity'].values

    array = xr.DataArray(
        data=intensity,
        coords={'energy': energy},
        attrs={
            'filename': filepath.name,
            'sample': utilities.get_sample_name(filepath.name),
            'spot': utilities.get_spot(filepath.name)
        },
        name='intensity'
    )
    array.energy.attrs['units'] = 'eV'

    return array


def bfield_contour_plot(data_directory : Path):
    
    bfield_file_list = utilities.get_file_paths(data_directory)
    spectra = []
    for filepath in sorted(bfield_file_list):
        df = pd.read_csv(filepath)                    # one spectrum
        field = utilities.get_field(filepath)          # reuse the existing helper
        da = xr.DataArray(
            df['Intensity'].values,
            coords={"wavelength": df['Wavelength'].values,
                    },
            dims=["wavelength"],
        )
        energy_values = const.h * const.c / (da['wavelength'].values * 1e-9) / const.e
        da = da.expand_dims(field=[field])              # tag this spectrum with its field
        da = da.assign_coords(
            energy=("wavelength", energy_values)
            )
        spectra.append(da)

    bfield_map = xr.concat(spectra, dim="field")       # stack into 2D
    bfield_map.plot.contourf(x="Magnetic Field (T)", y="Energy (eV)")    # contour map, axes auto-labeled
    plt.show()

def process_pl_directory_gaussian(experiment_directory) -> list:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    results = []

    for file in experiment_directory.iterdir():
        if file.is_file():
            try:
                data_array = load_pl_data(file)
                # maskcondition = (data_array.energy > 1.25) & (data_array.energy < 1.4)
                # da_masked = data_array.where(maskcondition, drop=True)
                fit_result = ff.fit_gaussian(data_array,
                                          inputpeaks=[1.352, 1.334],
                                          inputamps=[400, 80],
                                          inputwids=[0.01, 0.025],
                                          fixed_energies={},
                                          fixed_widths={})
                result = {
                    'sample': f'{data_array.attrs['sample']} {data_array.attrs['spot']}',
                    'peak1_energy': fit_result['peak_energies'][0] if fit_result['peak_energies'] else None,
                    'peak1_width': fit_result['peak_widths'][0] if fit_result['peak_widths'] else None,
                    'peak1_area' : fit_result['peak_areas'][0] if fit_result['peak_areas'] else None,
                    'peak2_energy': fit_result['peak_energies'][1] if fit_result['peak_energies'] else None,
                    'peak2_width': fit_result['peak_widths'][1] if fit_result['peak_widths'] else None,
                    'peak2_area' : fit_result['peak_areas'][1] if fit_result['peak_areas'] else None,
                    'area2/area1' : round(float(fit_result['peak_areas'][1]/fit_result['peak_areas'][0]), 3)
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

    return results

def plot_multiple_together(directory: Path) -> None:
    """
    Plots all PL spectra in a directory on a single figure.
    Applies an energy mask (1.26–1.38 eV) and Savitzky-Golay smoothing.
    """
    labellist = ['$MoS_2$ HS', '$WSe_2$ HS', 'Bare CrSBr']
    plt.figure(figsize=(6, 6), dpi=400)
    i = 0
    for file in directory.iterdir():
        if file.is_file():
            data_array = load_pl_data(file)
            maskcondition = (data_array.energy > 1.26) & (data_array.energy < 1.38)
            da_masked = data_array.where(maskcondition, drop=True)
            x = da_masked.energy.values
            y = da_masked.values
            y = smooth(y, 15, 3)
            plt.plot(x, y, label=f'{labellist[i]}')
            i += 1
    plt.legend()
    plt.xlabel('Energy (eV)', fontsize='large')
    plt.ylabel('PL Intensity (arb. units)', fontsize='x-large')
    plt.show()


def _safe_smooth(y, window=11, poly=3):
    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window <= poly:
        return y
    return smooth(y, window, poly)


def _fit_spectrum(data: xr.DataArray, peak_positions: list,
                  lineshape: str, energy_range: tuple) -> dict | None:
    """
    Fit N peaks to the spectrum within energy_range using clicked peak positions
    as initial guesses. Returns a result dict or None on failure.

    lineshape: 'gaussian' or 'lorentzian'
    energy_range: (xmin, xmax) in eV — typically the current axes zoom window.
    """
    xmin, xmax = energy_range
    mask = (data.energy >= xmin) & (data.energy <= xmax)
    da = data.where(mask, drop=True)

    if len(da) < 5:
        print("Too few data points in the current zoom window.")
        return None

    x = da.energy.values.astype(float)
    y = da.values.astype(float)
    y = _safe_smooth(y)
    y -= y.min()

    # Initial amplitude: look up smoothed value at each clicked position.
    inputamps = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x - pos)))
        inputamps.append(max(float(y[idx]), 1.0))

    inputwids = [0.01] * len(peak_positions)
    p0, bounds = ff.generateParams_fixed(peak_positions, inputamps, inputwids)

    diff_fn = ff.diffG if lineshape == 'gaussian' else ff.diffL
    fn      = ff.Gaussian if lineshape == 'gaussian' else ff.Lorentz

    try:
        result = minimize(diff_fn, p0, args=(x, y), bounds=bounds, method='L-BFGS-B')
        fit = result.x

        peaks = []
        for j in range(len(peak_positions)):
            amp    = abs(float(fit[3*j]))
            energy = float(fit[3*j + 1])
            width  = abs(float(fit[3*j + 2]))
            if lineshape == 'gaussian':
                sigma = width / (2 * np.sqrt(2 * np.log(2)))
                area  = amp * sigma * np.sqrt(2 * np.pi)
            else:
                area = amp * np.pi * width / 2
            peaks.append({
                'energy':    round(energy, 4),
                'width':     round(width, 4),
                'amplitude': round(amp, 2),
                'area':      round(area, 4),
                'y':         fn(fit[3*j:3*(j+1)], x),
            })

        return {'peaks': peaks, 'x_fit': x, 'y_total': fn(fit, x)}

    except Exception as e:
        print(f"Fit error: {e}")
        return None


class PLFitter:
    """
    Interactive matplotlib fitter for steady-state PL spectra.

    Left-click on the spectrum to place a peak marker.
    Right-click on a marker to remove it.
    Use the buttons to fit, save, and advance through files.

    IMPORTANT: Must be run as a plain script, not in a VS Code interactive
    (#%%) cell — the Jupyter kernel intercepts matplotlib click events.

    Usage:
        files = sorted(Path(cfg['pl_dir']).glob('2026-07-01/*.csv'))
        fitter = PLFitter(files, output_dir=Path(cfg['processed_pl_dir']))
        fitter.run()

    Results are appended to <output_dir>/fit_results.csv (one row per peak).
    Fit-overlay plots are saved to <output_dir>/plots/<filename>_fit.png.
    """

    def __init__(self, files: list, output_dir: Path):
        self.files = [Path(f) for f in files]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)

        self.current_idx = 0
        self.data        = None   # current xr.DataArray
        self.peak_markers = []    # [{'x': float, 'line': Artist, 'text': Artist}]
        self.fit_lines   = []     # artists for the current fit overlay
        self.fit_result  = None   # most recent _fit_spectrum result dict

        self.fig = None
        self.ax  = None

    def run(self):
        if not self.files:
            print("No files to process.")
            return

        self.fig = plt.figure(figsize=(10, 7))
        self.ax  = self.fig.add_axes([0.10, 0.18, 0.87, 0.77])

        ax_fitg  = self.fig.add_axes([0.03, 0.04, 0.16, 0.08])
        ax_fitl  = self.fig.add_axes([0.21, 0.04, 0.17, 0.08])
        ax_clear = self.fig.add_axes([0.40, 0.04, 0.14, 0.08])
        ax_save  = self.fig.add_axes([0.56, 0.04, 0.10, 0.08])
        ax_next  = self.fig.add_axes([0.68, 0.04, 0.10, 0.08])

        btn_fitg  = Button(ax_fitg,  'Fit Gaussian')
        btn_fitl  = Button(ax_fitl,  'Fit Lorentzian')
        btn_clear = Button(ax_clear, 'Clear Peaks')
        btn_save  = Button(ax_save,  'Save')
        btn_next  = Button(ax_next,  'Next →')

        btn_fitg.on_clicked(lambda _: self._fit('gaussian'))
        btn_fitl.on_clicked(lambda _: self._fit('lorentzian'))
        btn_clear.on_clicked(self._clear_peaks)
        btn_save.on_clicked(self._save)
        btn_next.on_clicked(self._next)

        # Keep references so buttons aren't garbage-collected.
        self._buttons = [btn_fitg, btn_fitl, btn_clear, btn_save, btn_next]

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._load_current()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: file loading
    # ------------------------------------------------------------------

    def _load_current(self):
        self.peak_markers.clear()
        self.fit_lines.clear()
        self.fit_result = None

        filepath = self.files[self.current_idx]
        self.data = load_pl_data(filepath)

        self.ax.cla()
        self.ax.plot(self.data.energy.values, self.data.values,
                     color='black', lw=1.2, label='Data')

        sample = self.data.attrs.get('sample', '')
        spot   = self.data.attrs.get('spot', '')
        n, total = self.current_idx + 1, len(self.files)
        self.ax.set_title(
            f'[{n}/{total}]  {filepath.name}\n'
            f'{sample}  {spot}  —  '
            'left-click: place peak  |  right-click: remove peak',
            fontsize=9
        )
        self.ax.set_xlabel('Energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal: click handling
    # ------------------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes is not self.ax:
            return
        if event.button == 1:
            self._add_peak(event.xdata)
        elif event.button == 3:
            self._remove_nearest_peak(event.xdata)

    def _add_peak(self, x):
        line = self.ax.axvline(x, color='steelblue', linestyle='--', alpha=0.8, lw=1.2)
        # Blended transform: data x, axes-fraction y — stays at top regardless of zoom.
        text = self.ax.text(
            x, 0.96, f'{x:.4f}',
            transform=self.ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color='steelblue', rotation=90,
        )
        self.peak_markers.append({'x': x, 'line': line, 'text': text})
        self.fig.canvas.draw_idle()

    def _remove_nearest_peak(self, x):
        if not self.peak_markers:
            return
        nearest = min(self.peak_markers, key=lambda m: abs(m['x'] - x))
        nearest['line'].remove()
        nearest['text'].remove()
        self.peak_markers.remove(nearest)
        self.fig.canvas.draw_idle()

    def _clear_peaks(self, _event):
        for m in self.peak_markers:
            m['line'].remove()
            m['text'].remove()
        self.peak_markers.clear()
        self.fig.canvas.draw_idle()

    def _clear_fit(self):
        for artist in self.fit_lines:
            artist.remove()
        self.fit_lines.clear()

    # ------------------------------------------------------------------
    # Internal: fitting
    # ------------------------------------------------------------------

    def _fit(self, lineshape: str):
        if not self.peak_markers:
            print("Place at least one peak marker before fitting.")
            return

        peak_positions = sorted(m['x'] for m in self.peak_markers)
        xmin, xmax = self.ax.get_xlim()

        result = _fit_spectrum(self.data, peak_positions, lineshape, (xmin, xmax))
        if result is None:
            print("Fit failed — try adjusting peak positions or zoom window.")
            return

        self.fit_result = {**result, 'lineshape': lineshape}

        self._clear_fit()

        colors = plt.cm.tab10.colors
        total_line, = self.ax.plot(result['x_fit'], result['y_total'],
                                   color='red', lw=1.5, label='Total fit')
        self.fit_lines.append(total_line)

        for j, peak in enumerate(result['peaks']):
            line, = self.ax.plot(
                result['x_fit'], peak['y'],
                linestyle='--', color=colors[j % 10], lw=1.2,
                label=f"Peak {j+1}: {peak['energy']:.4f} eV",
            )
            self.fit_lines.append(line)

        self.ax.legend(fontsize=8)
        self.fig.canvas.draw_idle()

        print(f"\nFit ({lineshape}):")
        for j, peak in enumerate(result['peaks']):
            print(f"  Peak {j+1}: E={peak['energy']:.4f} eV | "
                  f"FWHM={peak['width']:.4f} eV | "
                  f"amp={peak['amplitude']:.1f} | area={peak['area']:.4f}")

    # ------------------------------------------------------------------
    # Internal: save and navigation
    # ------------------------------------------------------------------

    def _save(self, _event):
        if self.fit_result is None:
            print("Run a fit before saving.")
            return

        filepath = self.files[self.current_idx]
        da = self.data

        rows = []
        for j, peak in enumerate(self.fit_result['peaks']):
            rows.append({
                'filename':    filepath.name,
                'sample':      da.attrs.get('sample', ''),
                'spot':        da.attrs.get('spot', ''),
                'field':       utilities.get_field(filepath),
                'temperature': utilities.get_temperature(filepath),
                'lineshape':   self.fit_result['lineshape'],
                'peak_index':  j,
                'energy':      peak['energy'],
                'width':       peak['width'],
                'amplitude':   peak['amplitude'],
                'area':        peak['area'],
            })

        csv_path = self.output_dir / 'fit_results.csv'
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
        print(f"Saved {len(rows)} peak(s) → {csv_path}")

        # Build a clean figure: no buttons, no peak markers.
        # Reconstruct the processed data (smoothed + baseline-subtracted) that
        # was actually fit, so data and overlay are on the same vertical scale.
        x_fit = self.fit_result['x_fit']
        mask = (da.energy >= x_fit.min()) & (da.energy <= x_fit.max())
        y_display = da.where(mask, drop=True).values.astype(float)
        y_display = _safe_smooth(y_display)
        y_display -= y_display.min()

        colors = plt.cm.tab10.colors
        fig_save, ax_save = plt.subplots(figsize=(8, 5))
        ax_save.plot(x_fit, y_display, color='black', lw=1.2, label='Data')
        ax_save.plot(x_fit, self.fit_result['y_total'],
                     color='red', lw=1.5, label='Total fit')
        for j, peak in enumerate(self.fit_result['peaks']):
            ax_save.plot(x_fit, peak['y'], linestyle='--',
                         color=colors[j % 10], lw=1.2,
                         label=f"Peak {j+1}: {peak['energy']:.4f} eV")
        ax_save.set_title(f"{filepath.name} fit")
        ax_save.set_xlabel('Energy (eV)')
        ax_save.set_ylabel('Intensity (arb. units)')
        ax_save.legend(fontsize=8)
        fig_save.tight_layout()

        plot_path = self.output_dir / 'plots' / (filepath.stem + '_fit.png')
        fig_save.savefig(plot_path, dpi=150)
        plt.close(fig_save)
        print(f"Plot → {plot_path}")

    def _next(self, _event):
        if self.current_idx + 1 >= len(self.files):
            print("No more files — closing.")
            plt.close(self.fig)
            return
        self.current_idx += 1
        self._clear_fit()
        self._load_current()


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def load_fit_results(csv_path: Path) -> pd.DataFrame:
    """Load fit_results.csv produced by PLFitter into a DataFrame."""
    return pd.read_csv(csv_path)


def plot_peak_param(df: pd.DataFrame, x_col: str, y_col: str,
                    peak_index: int | None = None,
                    group_by: str | None = None,
                    title: str = '') -> None:
    """
    Plot one fit parameter against another from fit_results.csv.

    x_col, y_col  : column names, e.g. 'field', 'temperature', 'energy', 'width'
    peak_index    : if given (0-based), filter to that peak before plotting
    group_by      : column to split into separate labelled series, e.g. 'sample'
    title         : axes title; auto-generated if omitted

    Examples:
        plot_peak_param(df, 'field', 'energy', peak_index=0, group_by='sample')
        plot_peak_param(df, 'temperature', 'width', peak_index=1)
    """
    data = df.copy()
    if peak_index is not None:
        data = data[data['peak_index'] == peak_index]

    fig, ax = plt.subplots(figsize=(7, 5))

    if group_by and group_by in data.columns:
        for label, group in data.groupby(group_by):
            group = group.sort_values(x_col)
            ax.plot(group[x_col], group[y_col], 'o-', label=str(label))
        ax.legend()
    else:
        data = data.sort_values(x_col)
        ax.plot(data[x_col], data[y_col], 'o-', color='steelblue')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)
    elif peak_index is not None:
        ax.set_title(f'Peak {peak_index + 1}: {y_col} vs {x_col}')
    else:
        ax.set_title(f'{y_col} vs {x_col}')
    plt.tight_layout()
    plt.show()


def process_pl_directory_lorentzian(experiment_directory) -> list:
    """
    Processes all CSV files in the PL directory for the given experiment date.
    Returns list of results dicts.
    """
    results = []

    for file in experiment_directory.iterdir():
        if file.is_file():
            try:
                data_array = load_pl_data(file)
                fit_result = ff.fit_lorentzian(data_array,
                             inputpeaks=[1.351, 1.327],
                             inputamps=[1000, 200],
                             inputwids=[0.025, 0.05],
                             fixed_energies={},
                             fixed_widths = {0 : 0.0125})
                result = {
                    'filename': file.name,
                    'sample': data_array.attrs['sample'],
                    'spot': data_array.attrs['spot'],
                    'peak1_energy': fit_result['peak_energies'][0] if fit_result['peak_energies'] else None,
                    'peak1_amplitude': fit_result['peak_amplitudes'][0] if fit_result['peak_amplitudes'] else None,
                    'peak2_energy': fit_result['peak_energies'][1] if fit_result['peak_energies'] else None,
                    'peak2_amplitude': fit_result['peak_amplitudes'][1] if fit_result['peak_amplitudes'] else None,
                    'fit_success': fit_result['fit_success']
                }
                results.append(result)
            except Exception as e:
                print(f"Error processing {file.name}: {e}")

    return results