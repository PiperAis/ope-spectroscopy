"""
Steady-state PL processing: loading, interactive fitting, and comparison.

Key components:
- load_pl_data: load a single PL CSV into an xr.DataArray (energy coords)
- PLFitter: interactive matplotlib fitter — click to place peaks, choose
  Gaussian or Lorentzian lineshape, save results to CSV + PNG
- load_fit_results / plot_peak_param: load fit_results.csv and plot any
  parameter pair across datasets (energy vs field, width vs temperature, etc.)
- BfieldContourBuilder: interactive heatmap builder for PL intensity vs
  energy and B-field — colormap, energy bounds, and normalization live,
  save to PNG when done
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import savgol_filter as smooth
from matplotlib.widgets import Button, TextBox, CheckButtons
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.lines import Line2D
from . import fitting_functions as ff
from . import utilities

xr.set_options(keep_attrs=True)

def load_pl_data(filepath: Path) -> xr.DataArray:
    """
    Loads PL CSV data into xarray DataArray.
    Assumes columns: Wavelength, Intensity.
    Converts wavelength to energy in eV. Returned sorted by ascending energy
    (wavelength-ordered CSVs would otherwise give descending energy).
    """
    
    if filepath.suffix == '.nc':
        array = xr.load_dataarray(filepath)
    else:
        data = pd.read_csv(filepath, delimiter=',')
        wavelength = data['Wavelength'].values
        energy = 1239.8 / wavelength # type: ignore
        intensity = data['Intensity'].values

        array = xr.DataArray(
            data=intensity,
            coords={'energy': energy},
            attrs={'filename': filepath.name},
            name='intensity'
        )
        array.energy.attrs['units'] = 'eV'

    # Resolve metadata (filename + folder metadata.txt) once, at load time.
    # setdefault: attrs already stored in a .nc file take precedence.
    metadata = utilities.get_metadata(filepath)
    for key in ('sample', 'spot', 'field', 'temperature'):
        array.attrs.setdefault(key, metadata[key])

    return array.sortby('energy')


def _load_bfield_map(files: Path | list[Path]) -> xr.DataArray:
    """
    Loads a set of PL spectra into one 2D (field, energy) DataArray.

    files: a directory (all files inside are used) or a list of PL spectrum
    CSVs, one per field value. Field for each is parsed from its filename
    via utilities.get_field. Sorted by field value (not acquisition order),
    since B-field sweeps commonly ramp up then back down through zero.
    """
    if isinstance(files, (str, Path)):
        files = [files]
    files = [Path(f) for f in files]
    if len(files) == 1 and files[0].is_dir():
        files = utilities.get_file_paths(files[0])

    spectra = []
    for filepath in sorted(files):
        da = load_pl_data(filepath)
        field = da.attrs['field']
        if field is None:
            warnings.warn(f"Could not parse magnetic field from filename or "
                          f"{utilities.METADATA_FILENAME}: {filepath.name}")
        spectra.append(da.expand_dims(field=[field]))

    bfield_map = xr.concat(spectra, dim="field").sortby("field")
    bfield_map.attrs.pop('field', None)  # per-spectrum scalar; now a coordinate
    bfield_map['field'].attrs = {'long_name': 'Magnetic Field', 'units': 'T'}
    bfield_map['energy'].attrs['long_name'] = 'Energy'
    return bfield_map


class _EnergyRangeControl:
    """
    E min / E max text-box pair shared by the interactive GUIs.

    Parses typed values, clamps them to the current data's energy range
    (replacing the box text with the data edge), and calls on_change() after
    a user edit. Programmatic box updates don't trigger on_change.
    """

    def __init__(self, ax_emin: Axes, ax_emax: Axes, on_change,
                 label_min: str = '', label_max: str = ''):
        self.tb_emin = TextBox(ax_emin, label_min)
        self.tb_emax = TextBox(ax_emax, label_max)
        self.tb_emin.on_submit(lambda text: self._on_submit(text, 'emin'))
        self.tb_emax.on_submit(lambda text: self._on_submit(text, 'emax'))
        self._on_change = on_change
        self.emin: float | None = None
        self.emax: float | None = None
        self._bounds = (-np.inf, np.inf)
        self._syncing = False

    @property
    def widgets(self) -> list:
        return [self.tb_emin, self.tb_emax]

    @property
    def range(self) -> tuple[float, float]:
        """Current (lo, hi), sorted so inverted entries still work."""
        assert self.emin is not None and self.emax is not None
        lo, hi = sorted((self.emin, self.emax))
        return lo, hi

    def set_data_bounds(self, lo: float, hi: float):
        """Set the clamp limits. First call: range = full data range.
        Later calls: keep the current range, clamped to the new limits."""
        self._bounds = (lo, hi)
        if self.emin is None or self.emax is None:
            self.emin, self.emax = lo, hi
        else:
            self.emin = float(np.clip(self.emin, lo, hi))
            self.emax = float(np.clip(self.emax, lo, hi))
        self._set_box(self.tb_emin, self.emin)
        self._set_box(self.tb_emax, self.emax)

    def _set_box(self, tb: TextBox, value: float):
        # set_val fires on_submit; suppress it for programmatic updates.
        self._syncing = True
        try:
            tb.set_val(f"{value:.4f}")
        finally:
            self._syncing = False

    def _on_submit(self, text: str, which: str):
        if self._syncing:
            return
        try:
            value = float(text)
        except ValueError:
            print(f"Invalid energy value: {text!r}")
            return
        clamped = float(np.clip(value, *self._bounds))
        tb = self.tb_emin if which == 'emin' else self.tb_emax
        if clamped != value:
            print(f"{'E min' if which == 'emin' else 'E max'} clamped to data edge: {clamped:.4f}")
            self._set_box(tb, clamped)
        if clamped == getattr(self, which):
            return
        setattr(self, which, clamped)
        self._on_change()


class BfieldContourBuilder:
    """
    Interactive contour (heatmap) builder for PL intensity vs. energy and
    magnetic field.

    Live controls: colormap, lower/upper energy bound, and a toggle to
    normalize all spectra against the dataset's global max. Save writes the
    current figure to a PNG.

    Usage:
        files = select_directory(Path(cfg['pl_dir']))
        BfieldContourBuilder(files, output_dir=Path(cfg['processed_pl_dir'])).run()
    """

    DEFAULT_CMAP = 'inferno'

    def __init__(self, files: Path | list[Path], output_dir: Path):
        self.data = _load_bfield_map(files)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cmap = self.DEFAULT_CMAP
        self.normalize = False

        self.fig: Figure  # created in run()
        self.erange: _EnergyRangeControl
        self.ax: Axes
        self._cbar: Colorbar | None = None

    def run(self):
        self.fig = plt.figure(figsize=(9, 9), layout="constrained")

        outer_gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(outer_gs[0])  # plot; colorbar auto-placed via ax=

        controls_gs = outer_gs[1].subgridspec(nrows=1, ncols=2, width_ratios=[3, 1])
        textbox_gs = controls_gs[0].subgridspec(nrows=1, ncols=3)
        side_gs = controls_gs[1].subgridspec(nrows=2, ncols=1, height_ratios=[2, 1])

        ax_cmap = self.fig.add_subplot(textbox_gs[0, 0])
        ax_emin = self.fig.add_subplot(textbox_gs[0, 1])
        ax_emax = self.fig.add_subplot(textbox_gs[0, 2])
        ax_norm = self.fig.add_subplot(side_gs[0, 0])
        ax_save = self.fig.add_subplot(side_gs[1, 0])

        self.tb_cmap = TextBox(ax_cmap, '', initial=self.cmap)
        self.erange = _EnergyRangeControl(ax_emin, ax_emax, on_change=self._draw)
        self.erange.set_data_bounds(float(self.data.energy.min()),
                                    float(self.data.energy.max()))
        ax_cmap.set_title('Colormap', fontsize=9)
        ax_emin.set_title('E min (eV)', fontsize=9)
        ax_emax.set_title('E max (eV)', fontsize=9)
        self.cb_norm = CheckButtons(ax_norm, ['Normalize to\nglobal max'], [False])
        self.btn_save = Button(ax_save, 'Save')

        self.tb_cmap.on_submit(self._on_cmap)
        self.cb_norm.on_clicked(self._on_normalize)
        self.btn_save.on_clicked(self._save)

        self._draw()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: compute and draw
    # ------------------------------------------------------------------

    def _current_data(self):
        lo, hi = self.erange.range
        cropped = self.data.sel(energy=slice(lo, hi))
        if self.normalize:
            cropped = cropped / float(self.data.max())
        return cropped

    def _draw(self):
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self.ax.cla()
        data = self._current_data()
        result = data.plot.contourf(x='field', y='energy', cmap=self.cmap,
                                     ax=self.ax, add_colorbar=False)
        self._cbar = self.fig.colorbar(result, ax=self.ax)
        self._cbar.set_label('Intensity (normalized)' if self.normalize else 'Intensity (arb. units)')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal: widget callbacks
    # ------------------------------------------------------------------

    def _on_cmap(self, text):
        if text not in plt.colormaps():
            print(f"Unknown colormap: {text!r}")
            return
        self.cmap = text
        self._draw()

    def _on_normalize(self, _label):
        self.normalize = not self.normalize
        self._draw()

    def _save(self, _event):
        sample = self.data.attrs.get('sample', 'bfield')
        out_path = self.output_dir / f"{sample.replace(' ', '')}_bfield_contour.png"

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        assert self._cbar is not None
        bbox = Bbox.union([  # pyright: ignore[reportArgumentType]
            self.ax.get_tightbbox(renderer),
            self._cbar.ax.get_tightbbox(renderer),
        ]).transformed(self.fig.dpi_scale_trans.inverted())

        self.fig.savefig(out_path, dpi=200, bbox_inches=bbox)
        print(f"Saved -> {out_path}")

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
            y = np.asarray(smooth(y, 15, 3))
            plt.plot(x, y, label=f'{labellist[i]}')
            i += 1
    plt.legend()
    plt.xlabel('Energy (eV)', fontsize='large')
    plt.ylabel('PL Intensity (arb. units)', fontsize='x-large')
    plt.show()


def _baseline_from_params(btype: str, bp, xc: float) -> dict:
    """Pack baseline parameters ([c0] or [c0, c1]) into a baseline dict."""
    return {'type': btype, 'c0': float(bp[0]),
            'c1': float(bp[1]) if len(bp) == 2 else 0.0, 'xc': xc}


def _eval_baseline(bl: dict, x):
    """Evaluate a baseline dict b(x) = c0 + c1·(x − xc)."""
    return bl['c0'] + bl['c1'] * (np.asarray(x) - bl['xc'])


def _fit_spectrum(data: xr.DataArray, peak_positions: list,
                  lineshape: str, energy_range: tuple,
                  baseline: str = 'constant') -> dict | None:
    """
    Fit N peaks to the spectrum within energy_range using clicked peak positions
    as initial guesses. Returns a result dict or None on failure.

    lineshape: 'gaussian' or 'lorentzian'
    energy_range: (xmin, xmax) in eV — the E min/E max range set in the GUI.
    baseline: 'constant' or 'linear' — fit jointly with the peaks as
              b(x) = c0 [+ c1·(x − xc)], xc = window center.
    """
    xmin, xmax = energy_range
    mask = (data.energy >= xmin) & (data.energy <= xmax)
    da = data.where(mask, drop=True)

    if len(da) < 5:
        print("Too few data points in the E min/E max range.")
        return None

    x = da.energy.values.astype(float)
    y = da.values.astype(float)

    # Baseline is fit as extra parameter(s) alongside the peaks.
    # Initial guess from the medians of the window's edge regions (robust to noise).
    xc = 0.5 * (float(x.min()) + float(x.max()))
    edge = max(1, len(y) // 10)
    xa, ya = float(np.mean(x[:edge])), float(np.median(y[:edge]))
    xb, yb = float(np.mean(x[-edge:])), float(np.median(y[-edge:]))
    if baseline == 'linear':
        c1 = (yb - ya) / (xb - xa) if xb != xa else 0.0
        c0 = ya + c1 * (xc - xa)
        bl0, bl_bounds = [c0, c1], [(float(y.min()), float(y.max())), (None, None)]
    else:
        c0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
        bl0, bl_bounds = [c0], [(float(y.min()), float(y.max()))]
    k = len(bl0)
    bl_guess = _baseline_from_params(baseline, bl0, xc)

    # Initial amplitude: data value above the baseline guess at each clicked position.
    inputamps = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x - pos)))
        inputamps.append(max(float(y[idx] - _eval_baseline(bl_guess, x[idx])), 1.0))

    inputwids = [0.01] * len(peak_positions)
    p0, bounds = ff.generateParams_fixed(peak_positions, inputamps, inputwids)
    p0 = list(p0) + bl0
    bounds = list(bounds) + bl_bounds

    fn, _ = ff.LINESHAPES[lineshape]

    # least_squares takes (lo, hi) arrays and needs lo < hi, so fixed parameters
    # (equal bounds from generateParams_fixed) aren't supported here; PLFitter never fixes any.
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)

    def residuals(p):
        return fn(p[:-k], x) + _eval_baseline(_baseline_from_params(baseline, p[-k:], xc), x) - y

    try:
        # Residual-vector solver with Jacobian scaling: parameters span very different
        # magnitudes (heights ~1e3, centers ~1 eV, widths ~1e-2), which made the old
        # scalar L-BFGS-B fit stop well short of the minimum.
        result = least_squares(residuals, p0, bounds=(lo, hi), x_scale='jac')
        bl = _baseline_from_params(baseline, result.x[-k:], xc)
        fit = result.x[:-k]
        y = y - _eval_baseline(bl, x)

        peaks = []
        for j in range(len(peak_positions)):
            amp    = abs(float(fit[3*j]))
            energy = float(fit[3*j + 1])
            width  = abs(float(fit[3*j + 2]))
            area   = ff.peak_area(lineshape, amp, width)
            peaks.append({
                'energy':    round(energy, 4),
                'width':     round(width, 4),
                'amplitude': round(amp, 2),
                'area':      round(area, 4),
                'y':         fn(fit[3*j:3*(j+1)], x),
            })

        return {'peaks': peaks, 'x_fit': x, 'y_total': fn(fit, x),
                'y_data': y, 'baseline': bl}

    except Exception as e:
        print(f"Fit error: {e}")
        return None


def _plot_fit_overlay(ax: Axes, result: dict) -> list:
    """Draw the total fit and each component peak on ax; return the created artists."""
    colors = plt.cm.tab10.colors  # pyright: ignore[reportAttributeAccessIssue]
    x = result['x_fit']
    total, = ax.plot(x, result['y_total'], color='red', lw=1.5, label='Total fit')
    artists = [total]
    for j, peak in enumerate(result['peaks']):
        line, = ax.plot(x, peak['y'], linestyle='--', color=colors[j % 10], lw=1.2,
                        label=f"Peak {j+1}: {peak['energy']:.4f} eV")
        artists.append(line)
    return artists


class PLFitter:
    """
    Interactive matplotlib fitter for steady-state PL spectra.

    Left-click on the spectrum to place a peak marker.
    Right-click on a marker to remove it.
    Set the fit window with the E min / E max boxes (persists across files;
    toolbar zoom is view-only and does not affect the fit).
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
        self.data: xr.DataArray   # current spectrum; set in _load_current()
        self.peak_markers = []    # [{'x': float, 'line': Artist, 'text': Artist}]
        self.fit_lines   = []     # artists for the current fit overlay
        self.fit_result  = None   # most recent _fit_spectrum result dict
        self.carry_peaks = False  # seed next file's markers from this file's peaks
        self.baseline = 'constant'  # 'constant' or 'linear'; persists across files

        self.fig: Figure  # created in run()
        self.ax: Axes
        self.erange: _EnergyRangeControl  # fit/display range; persists across files
        self._data_line: Line2D

    def run(self):
        if not self.files:
            print("No files to process.")
            return

        self.fig = plt.figure(figsize=(10, 7))
        self.ax  = self.fig.add_axes((0.10, 0.25, 0.87, 0.68))

        ax_emin = self.fig.add_axes((0.14, 0.135, 0.12, 0.05))
        ax_emax = self.fig.add_axes((0.40, 0.135, 0.12, 0.05))
        self.erange = _EnergyRangeControl(ax_emin, ax_emax, on_change=self._apply_range,
                                          label_min='E min (eV) ', label_max='E max (eV) ')

        ax_fitg  = self.fig.add_axes((0.03, 0.04, 0.16, 0.08))
        ax_fitl  = self.fig.add_axes((0.21, 0.04, 0.17, 0.08))
        ax_clear = self.fig.add_axes((0.40, 0.04, 0.14, 0.08))
        ax_save  = self.fig.add_axes((0.56, 0.04, 0.10, 0.08))
        ax_next  = self.fig.add_axes((0.68, 0.04, 0.10, 0.08))

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

        ax_carry = self.fig.add_axes((0.80, 0.03, 0.18, 0.10))
        cb_carry = CheckButtons(ax_carry, ['Carry peaks\nto next file'], [self.carry_peaks])
        cb_carry.on_clicked(self._on_carry)

        # Same row/height as the E boxes, so it can't overlap the button row.
        ax_base = self.fig.add_axes((0.62, 0.135, 0.22, 0.05))
        self.btn_base = Button(ax_base, self._baseline_label())
        self.btn_base.on_clicked(self._on_baseline)

        # Keep references so widgets aren't garbage-collected.
        self._buttons = [btn_fitg, btn_fitl, btn_clear, btn_save, btn_next, cb_carry,
                         *self.erange.widgets, self.btn_base]

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._load_current()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: file loading
    # ------------------------------------------------------------------

    def _load_current(self, seed_peaks: list[float] | None = None):
        self.peak_markers.clear()
        self.fit_lines.clear()
        self.fit_result = None

        filepath = self.files[self.current_idx]
        self.data = load_pl_data(filepath)

        # First file: full data range. Later files: keep the previous range,
        # clamped in case this file covers less energy.
        self.erange.set_data_bounds(float(self.data.energy.min()),
                                    float(self.data.energy.max()))

        self.ax.cla()
        cropped = self._current_data()
        self._data_line, = self.ax.plot(cropped.energy.values, cropped.values,
                                        color='black', lw=1.2, label='Data')
        self._set_view()

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

        for x in seed_peaks or []:
            self._add_peak(x)
        self.fig.canvas.draw_idle()

    def _on_carry(self, _label):
        self.carry_peaks = not self.carry_peaks

    _BASELINES = ('constant', 'linear')

    def _baseline_label(self) -> str:
        return f"Baseline: {self.baseline.capitalize()} ▾"

    def _on_baseline(self, _event):
        # Cycle to the next baseline type and relabel the button.
        i = self._BASELINES.index(self.baseline)
        self.baseline = self._BASELINES[(i + 1) % len(self._BASELINES)]
        self.btn_base.label.set_text(self._baseline_label())
        self.fig.canvas.draw_idle()
        # If a fit is showing, refit with the new baseline so the two can be compared.
        if self.fit_result is not None:
            self._fit(self.fit_result['lineshape'])

    # ------------------------------------------------------------------
    # Internal: energy range
    # ------------------------------------------------------------------

    def _current_data(self) -> xr.DataArray:
        lo, hi = self.erange.range
        return self.data.sel(energy=slice(lo, hi))

    def _set_view(self):
        # Explicit xlim so peak markers outside the range don't widen autoscale.
        self.ax.set_xlim(*self.erange.range)
        self.ax.relim()
        self.ax.autoscale_view(scalex=False)

    def _apply_range(self):
        # A fit (and its baseline) belongs to the old window, so drop it.
        self._clear_fit()
        self.fit_result = None
        cropped = self._current_data()
        self._data_line.set_data(cropped.energy.values, cropped.values)
        self._set_view()
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
        self._remove_marker(nearest)
        self.fig.canvas.draw_idle()

    def _clear_peaks(self, _event):
        for m in list(self.peak_markers):
            self._remove_marker(m)
        self.fig.canvas.draw_idle()

    def _remove_marker(self, marker: dict):
        marker['line'].remove()
        marker['text'].remove()
        self.peak_markers.remove(marker)

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
        result = _fit_spectrum(self.data, peak_positions, lineshape, self.erange.range,
                               baseline=self.baseline)
        if result is None:
            print("Fit failed — try adjusting peak positions or the E min/E max range.")
            return

        self.fit_result = {**result, 'lineshape': lineshape}

        self._clear_fit()

        # Show the data on the same baseline-subtracted scale the fit used.
        # Always computed from raw self.data, so refits don't stack baselines.
        cropped = self._current_data()
        self._data_line.set_ydata(
            cropped.values - _eval_baseline(result['baseline'], cropped.energy.values))
        self._set_view()

        self.fit_lines.extend(_plot_fit_overlay(self.ax, result))
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
                'field':       da.attrs.get('field'),
                'temperature': da.attrs.get('temperature'),
                'lineshape':   self.fit_result['lineshape'],
                'baseline':    self.fit_result['baseline']['type'],
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
        # Plot exactly the (baseline-subtracted) data that was fit.
        fig_save, ax_save = plt.subplots(figsize=(8, 5))
        ax_save.plot(self.fit_result['x_fit'], self.fit_result['y_data'],
                     color='black', lw=1.2, label='Data')
        _plot_fit_overlay(ax_save, self.fit_result)
        ax_save.set_title(f"{filepath.name} fit ({self.fit_result['baseline']['type']} baseline)")
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
        seeds = None
        if self.carry_peaks:
            if self.fit_result is not None:
                seeds = [p['energy'] for p in self.fit_result['peaks']]
            else:
                seeds = [m['x'] for m in self.peak_markers]

        self.current_idx += 1
        self._clear_fit()
        self._load_current(seed_peaks=seeds)


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