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

def load_pl_data(filepath: Path, variable: str = 'raw') -> xr.DataArray:
    """
    Loads one PL spectrum into an xarray DataArray (energy coords, ascending).

    CSV (raw data): columns Wavelength, Intensity; wavelength converted to eV.
    Only variable='raw' is available from a CSV.

    .nc (processed, written by PLFitter on Save): a Dataset with variables
    'raw', 'baseline', 'subtracted'; `variable` picks one, and the Dataset's
    attrs (metadata + fit settings) are carried onto the returned array.
    A single-variable .nc (older format) is returned for variable='raw'.
    """
    filepath = Path(filepath)
    if filepath.suffix == '.nc':
        ds = xr.load_dataset(filepath)
        if variable in ds.data_vars:
            array = ds[variable]
        elif variable == 'raw' and len(ds.data_vars) == 1:
            array = ds[next(iter(ds.data_vars))]
        else:
            raise KeyError(f"{filepath.name} has no '{variable}' variable "
                           f"(has: {list(ds.data_vars)})")
        array = array.assign_attrs({**ds.attrs, **array.attrs})
    elif variable != 'raw':
        raise KeyError(f"{filepath.name} is raw data; '{variable}' is only available "
                       f"in a processed .nc (fit and Save it in PLFitter first)")
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


_SPECTRUM_SUFFIXES = ('.csv', '.nc')


def _spectrum_paths(files: Path | list[Path]) -> list[Path]:
    """A directory (its spectrum files) or an explicit list → sorted spectrum paths.
    Non-spectrum files in a directory (e.g. metadata.txt) are skipped."""
    if isinstance(files, (str, Path)):
        files = [files]
    files = [Path(f) for f in files]
    if len(files) == 1 and files[0].is_dir():
        files = [f for f in utilities.get_file_paths(files[0])
                 if f.suffix.lower() in _SPECTRUM_SUFFIXES]
    return sorted(files)


def _has_variable(filepath: Path, variable: str) -> bool:
    """True if filepath is a processed .nc containing `variable`."""
    if filepath.suffix != '.nc':
        return variable == 'raw'
    with xr.open_dataset(filepath) as ds:
        return variable in ds.data_vars


def _load_bfield_map(files: Path | list[Path], variable: str = 'raw') -> xr.DataArray:
    """
    Loads a set of PL spectra into one 2D (field, energy) DataArray.

    files: a directory (its CSV / processed .nc files are used) or an explicit
    list, one spectrum per field value. variable: 'raw', or 'subtracted' /
    'baseline' from processed .nc files. Field for each comes from attrs
    (filename / metadata.txt). Sorted by field value (not acquisition order),
    since B-field sweeps commonly ramp up then back down through zero.

    For processed spectra, attrs fit_emin/fit_emax on the map are the
    intersection of the individual fit windows.
    """
    spectra, windows = [], []
    for filepath in _spectrum_paths(files):
        da = load_pl_data(filepath, variable=variable)
        field = da.attrs['field']
        if field is None:
            warnings.warn(f"Could not parse magnetic field from filename or "
                          f"{utilities.METADATA_FILENAME}: {filepath.name}")
        if 'fit_emin' in da.attrs:
            windows.append((da.attrs['fit_emin'], da.attrs['fit_emax']))
        spectra.append(da.expand_dims(field=[field]))

    bfield_map = xr.concat(spectra, dim="field").sortby("field")
    # Drop per-spectrum attrs (now a coordinate, or specific to one fit).
    for key in list(bfield_map.attrs):
        if key in ('field', 'filename', 'lineshape', 'fit_emin', 'fit_emax',
                   'fit_number', 'rmse', 'r_squared') \
                or key.startswith(('peak_', 'baseline_')):
            del bfield_map.attrs[key]
    if windows and len(windows) == len(spectra):
        bfield_map.attrs['fit_emin'] = max(w[0] for w in windows)
        bfield_map.attrs['fit_emax'] = min(w[1] for w in windows)
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

    def set_range(self, lo: float, hi: float):
        """Programmatically set the range (clamped to the data bounds); no on_change."""
        self.emin = float(np.clip(lo, *self._bounds))
        self.emax = float(np.clip(hi, *self._bounds))
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


def _savgol_along(da: xr.DataArray, dim: str, window: int, polyorder: int) -> xr.DataArray:
    """Savitzky–Golay smooth `da` along `dim` (window in points; shrunk to fit if too long)."""
    n = da.sizes[dim]
    window = min(window, n if n % 2 else n - 1)
    if window <= polyorder:
        return da
    return xr.apply_ufunc(lambda a: smooth(a, window, polyorder, axis=-1), da,
                          input_core_dims=[[dim]], output_core_dims=[[dim]], keep_attrs=True)


class BfieldContourBuilder:
    """
    Interactive contour (heatmap) builder for PL intensity vs. energy and
    magnetic field.

    Live controls: colormap, lower/upper energy bound, a "Normalize" button
    that cycles Off → Global max (all spectra against the dataset's max) →
    Per spectrum (each spectrum against its own max within the E window), and an optional
    Savitzky–Golay smoothing along energy (window in points; display only,
    the underlying data is not changed). Save writes the current figure to a PNG.

    Input is a raw sweep folder (CSVs) or a processed sweep folder (.nc files
    written by PLFitter's Save). If every spectrum has been fitted, a
    "Data" button switches between baseline-subtracted (default) and raw.

    Usage:
        files = select_directory(Path(cfg['processed_pl_dir']))
        BfieldContourBuilder(files, output_dir=Path(cfg['processed_pl_dir'])).run()
    """

    DEFAULT_CMAP = 'inferno'
    _VARIABLES = ('subtracted', 'raw')
    SMOOTH_POLYORDER = 3
    DEFAULT_SMOOTH_WINDOW = 11   # points along energy; must be odd and > polyorder

    def __init__(self, files: Path | list[Path], output_dir: Path):
        paths = _spectrum_paths(files)
        self.maps = {'raw': _load_bfield_map(paths, 'raw')}
        unfitted = [p.name for p in paths if not _has_variable(p, 'subtracted')]
        if not unfitted:
            self.maps['subtracted'] = _load_bfield_map(paths, 'subtracted')
        elif any(p.suffix == '.nc' for p in paths):
            warnings.warn(f"{len(unfitted)} spectra have no baseline-subtracted data "
                          f"(not fitted/saved yet), showing raw only: {', '.join(unfitted)}")
        self.variable = 'subtracted' if 'subtracted' in self.maps else 'raw'

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cmap = self.DEFAULT_CMAP
        self.normalize = 'off'   # one of _NORMS
        self.smooth = False
        self.smooth_window = self.DEFAULT_SMOOTH_WINDOW

        self.fig: Figure  # created in run()
        self.erange: _EnergyRangeControl
        self.ax: Axes
        self._cbar: Colorbar | None = None

    def run(self):
        self.fig = plt.figure(figsize=(9, 9), layout="constrained")

        outer_gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(outer_gs[0])  # plot; colorbar auto-placed via ax=

        has_toggle = len(self.maps) > 1
        controls_gs = outer_gs[1].subgridspec(nrows=1, ncols=2, width_ratios=[3, 1])
        textbox_gs = controls_gs[0].subgridspec(nrows=1, ncols=4)
        side_gs = controls_gs[1].subgridspec(nrows=4 if has_toggle else 3, ncols=1)

        ax_cmap = self.fig.add_subplot(textbox_gs[0, 0])
        ax_emin = self.fig.add_subplot(textbox_gs[0, 1])
        ax_emax = self.fig.add_subplot(textbox_gs[0, 2])
        ax_win  = self.fig.add_subplot(textbox_gs[0, 3])
        ax_smooth = self.fig.add_subplot(side_gs[0, 0])
        ax_norm = self.fig.add_subplot(side_gs[1, 0])
        ax_save = self.fig.add_subplot(side_gs[-1, 0])

        self.tb_cmap = TextBox(ax_cmap, '', initial=self.cmap)
        self.erange = _EnergyRangeControl(ax_emin, ax_emax, on_change=self._draw)
        self.erange.set_data_bounds(float(self.data.energy.min()),
                                    float(self.data.energy.max()))
        if 'fit_emin' in self.data.attrs:
            # Open on the fitted window, not on extrapolated baseline.
            self.erange.set_range(self.data.attrs['fit_emin'], self.data.attrs['fit_emax'])
        ax_cmap.set_title('Colormap', fontsize=9)
        ax_emin.set_title('E min (eV)', fontsize=9)
        ax_emax.set_title('E max (eV)', fontsize=9)
        self.tb_win = TextBox(ax_win, '', initial=str(self.smooth_window))
        ax_win.set_title('Smooth window (pts)', fontsize=9)
        self.tb_win.on_submit(self._on_window)
        self.cb_smooth = CheckButtons(ax_smooth, [self._SMOOTH_LABEL], [self.smooth])
        self.btn_norm = Button(ax_norm, self._norm_label())
        self.btn_save = Button(ax_save, 'Save')
        if has_toggle:
            self.btn_data = Button(self.fig.add_subplot(side_gs[2, 0]), self._data_label())
            self.btn_data.on_clicked(self._on_data)

        self.tb_cmap.on_submit(self._on_cmap)
        self.cb_smooth.on_clicked(self._on_smooth)
        self.btn_norm.on_clicked(self._on_normalize)
        self.btn_save.on_clicked(self._save)

        self._draw()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: compute and draw
    # ------------------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        """The (field, energy) map currently shown: raw or baseline-subtracted."""
        return self.maps[self.variable]

    def _data_label(self) -> str:
        return f"Data: {self.variable.capitalize()} ▾"

    def _on_data(self, _event):
        # Cycle raw ↔ subtracted; keep the E range (clamped to the new map).
        options = [v for v in self._VARIABLES if v in self.maps]
        self.variable = options[(options.index(self.variable) + 1) % len(options)]
        self.btn_data.label.set_text(self._data_label())
        self.erange.set_data_bounds(float(self.data.energy.min()),
                                    float(self.data.energy.max()))
        self._draw()

    def _current_data(self):
        data = self.data
        if self.smooth:
            # Smooth the full spectra before cropping, so the E min/E max edges
            # don't pick up filter edge effects.
            data = _savgol_along(data, 'energy', self.smooth_window, self.SMOOTH_POLYORDER)
        lo, hi = self.erange.range
        cropped = data.sel(energy=slice(lo, hi))
        if self.normalize == 'global':
            cropped = cropped / float(data.max())
        elif self.normalize == 'spectrum':
            cropped = cropped / cropped.max('energy')
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
        label = 'Intensity'
        if self.variable == 'subtracted':
            label += ', baseline-subtracted'
        label += {'off': ' (arb. units)',
                  'global': ' (normalized)',
                  'spectrum': ' (normalized per spectrum)'}[self.normalize]
        if self.smooth:
            label += f'\nS-G smoothed, {self.smooth_window} pts'
        self._cbar.set_label(label)
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

    _SMOOTH_LABEL = 'Smooth (S-G)'
    _NORMS = ('off', 'global', 'spectrum')
    _NORM_NAMES = {'off': 'Off', 'global': 'Global max', 'spectrum': 'Per spectrum'}

    def _norm_label(self) -> str:
        return f"Normalize: {self._NORM_NAMES[self.normalize]} ▾"

    def _on_normalize(self, _event):
        # Cycle to the next normalization mode and relabel the button.
        i = self._NORMS.index(self.normalize)
        self.normalize = self._NORMS[(i + 1) % len(self._NORMS)]
        self.btn_norm.label.set_text(self._norm_label())
        self._draw()

    def _on_smooth(self, _label):
        self.smooth = not self.smooth
        self._draw()

    def _on_window(self, text):
        try:
            value = int(float(text))
        except ValueError:
            print(f"Invalid smoothing window: {text!r}")
            self.tb_win.set_val(str(self.smooth_window))   # restore the box
            return
        # Savitzky–Golay needs an odd window longer than the polynomial order.
        window = max(value, self.SMOOTH_POLYORDER + 2)
        if window % 2 == 0:
            window += 1
        if str(window) != text.strip():
            print(f"Smoothing window set to {window} (must be odd and > {self.SMOOTH_POLYORDER})")
            self.tb_win.set_val(str(window))   # re-submits with the valid value
            return
        if window == self.smooth_window:
            return
        self.smooth_window = window
        if self.smooth:
            self._draw()

    def _save(self, _event):
        sample = self.data.attrs.get('sample', 'bfield')
        suffix = '_subtracted' if self.variable == 'subtracted' else ''
        if self.smooth:
            suffix += f'_sg{self.smooth_window}'
        out_path = self.output_dir / f"{sample.replace(' ', '')}_bfield_contour{suffix}.png"

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
        y_sub = y - _eval_baseline(bl, x)   # new name: `y` stays the raw data residuals() uses

        # 1σ uncertainties from the Jacobian (residual variance estimated from the fit).
        cov = ff.param_covariance(result.jac, result.fun)
        err = np.sqrt(np.clip(np.diag(cov), 0, None))

        peaks = []
        for j in range(len(peak_positions)):
            ia, ie, iw = 3*j, 3*j + 1, 3*j + 2
            amp    = abs(float(fit[ia]))
            energy = float(fit[ie])
            width  = abs(float(fit[iw]))
            area   = ff.peak_area(lineshape, amp, width)
            # area ∝ amp·width for both lineshapes → propagate with the amp–width covariance.
            rel_var = ((err[ia] / amp) ** 2 + (err[iw] / width) ** 2
                       + 2 * cov[ia, iw] / (amp * width)) if amp > 0 and width > 0 else np.nan
            area_err = area * float(np.sqrt(max(rel_var, 0.0)))
            # Stored unrounded (energy uncertainty can be ~1e-5 eV); round only for display.
            peaks.append({
                'energy':        energy,
                'energy_err':    float(err[ie]),
                'width':         width,
                'width_err':     float(err[iw]),
                'amplitude':     amp,
                'amplitude_err': float(err[ia]),
                'area':          float(area),
                'area_err':      area_err,
                'y':             fn(fit[3*j:3*(j+1)], x),
            })

        # Fit quality from the same residuals (no per-point σ, so RMSE + R², not χ²).
        n_points = len(y)
        dof = n_points - len(p0)
        sse = float(np.sum(result.fun ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        rmse = float(np.sqrt(sse / dof)) if dof > 0 else np.nan
        r_squared = 1 - sse / sst if dof > 0 and sst > 0 else np.nan

        return {'peaks': peaks, 'x_fit': x, 'y_total': fn(fit, x),
                'y_data': y_sub, 'baseline': bl,
                'n_points': n_points, 'dof': dof, 'rmse': rmse, 'r_squared': r_squared}

    except Exception as e:
        print(f"Fit error: {e}")
        return None


def _append_rows_csv(csv_path: Path, df: pd.DataFrame):
    """
    Append rows to a CSV. If the existing file has different columns (an older
    schema), merge by column name and rewrite it rather than appending rows that
    don't line up with the header; columns missing on either side become blank.
    """
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        return
    existing_cols = list(pd.read_csv(csv_path, nrows=0).columns)
    if existing_cols == list(df.columns):
        df.to_csv(csv_path, mode='a', header=False, index=False)
        return
    merged = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
    merged = merged[list(df.columns) + [c for c in merged.columns if c not in df.columns]]
    merged.to_csv(csv_path, index=False)
    print(f"Note: {csv_path.name} had an older column layout; merged and rewrote it "
          f"(older rows have blank values in new columns).")


def _row_keys(frame: pd.DataFrame, keys: list[str]) -> pd.MultiIndex:
    """Comparable row keys. A CSV round trip can change a value's type (1 → 1.0
    when the column has blanks; folder name '20261105' → int), so numeric-looking
    values are compared as numbers and everything else as strings."""
    cols = {}
    for k in keys:
        num = pd.to_numeric(frame[k], errors='coerce').astype(float)
        cols[k] = num.astype(str).where(num.notna(), frame[k].astype(str))
    return pd.MultiIndex.from_frame(pd.DataFrame(cols))


def _upsert_rows_csv(csv_path: Path, df: pd.DataFrame, keys: list[str]):
    """
    Write rows to a CSV, replacing any existing rows whose `keys` values match
    a row in df (e.g. a re-saved fit). Other rows are kept. Columns are merged
    by name as in _append_rows_csv: df's columns first, extras after, and
    columns missing on either side become blank.
    """
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        return
    existing = pd.read_csv(csv_path)
    if list(existing.columns) != list(df.columns):
        print(f"Note: {csv_path.name} had an older column layout; merged and rewrote it "
              f"(older rows have blank values in new columns).")
    if all(k in existing.columns for k in keys):
        existing = existing[~_row_keys(existing, keys).isin(_row_keys(df, keys))]
    merged = pd.concat([existing, df], ignore_index=True)
    merged = merged[list(df.columns) + [c for c in merged.columns if c not in df.columns]]
    merged.to_csv(csv_path, index=False)


_PEAK_KEYS = ('energy', 'energy_err', 'width', 'width_err',
              'amplitude', 'amplitude_err', 'area', 'area_err')


def _save_processed_spectrum(path: Path, raw: xr.DataArray, result: dict, lineshape: str,
                             experiment: str, fit_number: int):
    """
    Write one fitted spectrum as a processed .nc (the PL analog of a TRR dataset
    file): variables 'raw', 'baseline' (fitted baseline evaluated over the full
    energy range) and 'subtracted' (raw − baseline). Metadata, fit settings and
    per-peak results go in the Dataset attrs. Overwrites, so a refit replaces it.

    Outside [fit_emin, fit_emax] the baseline is extrapolated from the fit window.
    """
    bl = result['baseline']
    metadata = {k: raw.attrs.get(k) for k in ('filename', 'sample', 'spot', 'field', 'temperature')}
    raw = raw.copy()
    raw.attrs = {}
    baseline = xr.DataArray(_eval_baseline(bl, raw.energy.values),
                            coords={'energy': raw.energy}, name='baseline')
    ds = xr.Dataset({'raw': raw, 'baseline': baseline, 'subtracted': raw - baseline})

    attrs = {k: v for k, v in metadata.items() if v is not None}
    attrs.update({
        'experiment':    experiment,
        'fit_number':    int(fit_number),   # matches plots/fits_<n>/ and the CSV
        'rmse':          float(result['rmse']),
        'r_squared':     float(result['r_squared']),
        'lineshape':     lineshape,
        'baseline_type': bl['type'],
        'baseline_c0':   bl['c0'],
        'baseline_c1':   bl['c1'],
        'baseline_xc':   bl['xc'],
        'fit_emin':      float(np.min(result['x_fit'])),
        'fit_emax':      float(np.max(result['x_fit'])),
    })
    for key in _PEAK_KEYS:   # one entry per peak, in peak order
        attrs[f'peak_{key}'] = np.array([p[key] for p in result['peaks']], dtype=float)
    ds.attrs = attrs   # netCDF can't store None attrs; those were dropped above

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, mode='w')


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
    Use the buttons to fit, save, and move through files. "← Back" (and Next
    onto an already-visited file) restores that file's markers, fit, range and
    baseline, so a fit can be saved or adjusted later. The title shows whether
    the current fit has been saved.

    IMPORTANT: Must be run as a plain script, not in a VS Code interactive
    (#%%) cell — the Jupyter kernel intercepts matplotlib click events.

    Usage:
        fitter = PLFitter(sweep_folder_or_file_list, output_dir=Path(cfg['processed_pl_dir']))
        fitter.run()

    Results go to <output_dir>/fit_results.csv: one row per peak, tagged with
    experiment and fit_number (= <n> below), plus the fit window and fit
    quality (rmse, r_squared). Re-saving a spectrum in the same session
    replaces its rows; earlier sessions' rows are kept.
    Fit-overlay plots are saved to
    <output_dir>/<experiment>/plots/fits_<n>/<filename>_fit.png, where
    <experiment> is the spectrum's sweep folder name and <n> counts up once
    per fitting session, so earlier iterations are never overwritten.
    """

    def __init__(self, files: Path | list[Path], output_dir: Path):
        self.files = _spectrum_paths(files)   # a folder (its .csv/.nc) or a list
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._fit_numbers: dict[str, int] = {}  # experiment name → this session's n (fits_<n>/)

        self.current_idx = 0
        self.data: xr.DataArray   # current spectrum; set in _load_current()
        self.peak_markers = []    # [{'x': float, 'line': Artist, 'text': Artist}]
        self.fit_lines   = []     # artists for the current fit overlay
        self.fit_result  = None   # most recent _fit_spectrum result dict
        self._saved      = False  # has the current fit_result been saved?
        self._states: dict[int, dict] = {}  # per-file snapshot for Back / revisits
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

        ax_fitg  = self.fig.add_axes((0.02, 0.04, 0.14, 0.08))
        ax_fitl  = self.fig.add_axes((0.17, 0.04, 0.15, 0.08))
        ax_clear = self.fig.add_axes((0.33, 0.04, 0.12, 0.08))
        ax_save  = self.fig.add_axes((0.46, 0.04, 0.09, 0.08))
        ax_back  = self.fig.add_axes((0.56, 0.04, 0.10, 0.08))
        ax_next  = self.fig.add_axes((0.67, 0.04, 0.10, 0.08))

        btn_fitg  = Button(ax_fitg,  'Fit Gaussian')
        btn_fitl  = Button(ax_fitl,  'Fit Lorentzian')
        btn_clear = Button(ax_clear, 'Clear Peaks')
        btn_save  = Button(ax_save,  'Save')
        btn_back  = Button(ax_back,  '← Back')
        btn_next  = Button(ax_next,  'Next →')

        btn_fitg.on_clicked(lambda _: self._fit('gaussian'))
        btn_fitl.on_clicked(lambda _: self._fit('lorentzian'))
        btn_clear.on_clicked(self._clear_peaks)
        btn_save.on_clicked(self._save)
        btn_back.on_clicked(self._back)
        btn_next.on_clicked(self._next)

        ax_carry = self.fig.add_axes((0.79, 0.03, 0.19, 0.10))
        cb_carry = CheckButtons(ax_carry, ['Carry peaks\nto next file'], [self.carry_peaks])
        cb_carry.on_clicked(self._on_carry)

        # Same row/height as the E boxes, so it can't overlap the button row.
        ax_base = self.fig.add_axes((0.62, 0.135, 0.22, 0.05))
        self.btn_base = Button(ax_base, self._baseline_label())
        self.btn_base.on_clicked(self._on_baseline)

        # Keep references so widgets aren't garbage-collected.
        self._buttons = [btn_fitg, btn_fitl, btn_clear, btn_save, btn_back, btn_next, cb_carry,
                         *self.erange.widgets, self.btn_base]

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._load_current()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: file loading
    # ------------------------------------------------------------------

    def _load_current(self, seed_peaks: list[float] | None = None,
                      state: dict | None = None):
        """Load files[current_idx]. `state` (a snapshot from _stash_state) restores a
        previously visited file; otherwise `seed_peaks` (carried peaks) are placed."""
        self.peak_markers.clear()
        self.fit_lines.clear()
        self.fit_result = None
        self._saved = False

        filepath = self.files[self.current_idx]
        self.data = load_pl_data(filepath)

        # First file: full data range. Later files: keep the previous range,
        # clamped in case this file covers less energy.
        self.erange.set_data_bounds(float(self.data.energy.min()),
                                    float(self.data.energy.max()))
        if state is not None:
            self.erange.set_range(*state['range'])
            self.baseline = state['baseline']
            self.btn_base.label.set_text(self._baseline_label())

        self.ax.cla()
        cropped = self._current_data()
        self._data_line, = self.ax.plot(cropped.energy.values, cropped.values,
                                        color='black', lw=1.2, label='Data')
        self._set_view()
        self.ax.set_xlabel('Energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')

        markers = state['markers'] if state is not None else (seed_peaks or [])
        for x in markers:
            self._add_peak(x)
        if state is not None and state['fit_result'] is not None:
            self.fit_result = state['fit_result']
            self._saved = state['saved']
            self._show_fit()
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        filepath = self.files[self.current_idx]
        sample = self.data.attrs.get('sample', '')
        spot   = self.data.attrs.get('spot', '')
        n, total = self.current_idx + 1, len(self.files)
        if self.fit_result is None:
            status = ''
        else:
            status = '   ✓ saved' if self._saved else '   (fit not saved)'
        self.ax.set_title(
            f'[{n}/{total}]  {filepath.name}{status}\n'
            f'{sample}  {spot}  —  '
            'left-click: place peak  |  right-click: remove peak',
            fontsize=9
        )

    def _stash_state(self):
        """Snapshot the current file so Back / revisiting restores it."""
        self._states[self.current_idx] = {
            'markers':    [m['x'] for m in self.peak_markers],
            'fit_result': self.fit_result,
            'saved':      self._saved,
            'range':      (self.erange.emin, self.erange.emax),
            'baseline':   self.baseline,
        }
        if self.fit_result is not None and not self._saved:
            print(f"Note: fit for {self.files[self.current_idx].name} is not saved "
                  f"(use ← Back to return to it).")

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
        self._saved = False
        cropped = self._current_data()
        self._data_line.set_data(cropped.energy.values, cropped.values)
        self._set_view()
        self._update_title()
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
        self._saved = False
        self._show_fit()

        print(f"\nFit ({lineshape}): RMSE={result['rmse']:.3g}, R²={result['r_squared']:.4f}")
        for j, peak in enumerate(result['peaks']):
            print(f"  Peak {j+1}: E={peak['energy']:.4f} ± {peak['energy_err']:.2g} eV | "
                  f"FWHM={peak['width']:.4f} ± {peak['width_err']:.2g} eV | "
                  f"amp={peak['amplitude']:.1f} ± {peak['amplitude_err']:.2g} | "
                  f"area={peak['area']:.4f} ± {peak['area_err']:.2g}")

    def _show_fit(self):
        """Draw self.fit_result: baseline-subtracted data plus the fit overlay."""
        result = self.fit_result
        assert result is not None
        self._clear_fit()

        # Show the data on the same baseline-subtracted scale the fit used.
        # Always computed from raw self.data, so refits don't stack baselines.
        cropped = self._current_data()
        self._data_line.set_ydata(
            cropped.values - _eval_baseline(result['baseline'], cropped.energy.values))
        self._set_view()

        self.fit_lines.extend(_plot_fit_overlay(self.ax, result))
        self.ax.legend(fontsize=8)
        self._update_title()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal: save and navigation
    # ------------------------------------------------------------------

    def _save(self, _event):
        if self.fit_result is None:
            print("Run a fit before saving.")
            return

        filepath = self.files[self.current_idx]
        da = self.data
        result = self.fit_result
        experiment = filepath.parent.name
        fit_number = self._fit_number(experiment)

        rows = []
        for j, peak in enumerate(result['peaks']):
            rows.append({
                'filename':    filepath.name,
                'experiment':  experiment,
                'fit_number':  fit_number,
                'sample':      da.attrs.get('sample', ''),
                'spot':        da.attrs.get('spot', ''),
                'field':       da.attrs.get('field'),
                'temperature': da.attrs.get('temperature'),
                'lineshape':   result['lineshape'],
                'baseline':    result['baseline']['type'],
                'fit_emin':    float(np.min(result['x_fit'])),
                'fit_emax':    float(np.max(result['x_fit'])),
                'peak_index':  j,
                'energy':        peak['energy'],
                'energy_err':    peak['energy_err'],
                'width':         peak['width'],
                'width_err':     peak['width_err'],
                'amplitude':     peak['amplitude'],
                'amplitude_err': peak['amplitude_err'],
                'area':          peak['area'],
                'area_err':      peak['area_err'],
                'n_points':    result['n_points'],
                'dof':         result['dof'],
                'rmse':        result['rmse'],
                'r_squared':   result['r_squared'],
            })

        # One row set per (spectrum, fit session): re-saving in this session replaces it.
        csv_path = self.output_dir / 'fit_results.csv'
        _upsert_rows_csv(csv_path, pd.DataFrame(rows),
                         keys=['experiment', 'filename', 'fit_number'])
        print(f"Saved {len(rows)} peak(s) as fit {fit_number} → {csv_path}")

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

        plot_path = self._fits_dir(experiment) / (filepath.stem + '_fit.png')
        fig_save.savefig(plot_path, dpi=150)
        plt.close(fig_save)
        print(f"Plot → {plot_path}")

        # Processed spectrum (raw / baseline / subtracted) for the contour builder,
        # one .nc per spectrum under <output_dir>/<sweep folder>/, like TRR's per-set .nc.
        nc_path = self.output_dir / experiment / f"{filepath.stem}.nc"
        _save_processed_spectrum(nc_path, self.data, self.fit_result,
                                 self.fit_result['lineshape'], experiment, fit_number)
        print(f"Spectrum → {nc_path}")

        self._saved = True
        self._update_title()
        self.fig.canvas.draw_idle()

    def _fits_dir(self, experiment: str) -> Path:
        """This session's <output_dir>/<experiment>/plots/fits_<n>/ folder.

        Chosen on the first save for that experiment as one past the highest
        existing fits_<n>; later saves in the same session reuse it.
        """
        plots_dir = self.output_dir / experiment / 'plots'
        if experiment not in self._fit_numbers:
            plots_dir.mkdir(parents=True, exist_ok=True)
            taken = [int(p.name[5:]) for p in plots_dir.glob('fits_*')
                     if p.is_dir() and p.name[5:].isdigit()]
            n = max(taken, default=0) + 1
            (plots_dir / f"fits_{n}").mkdir()
            self._fit_numbers[experiment] = n
        return plots_dir / f"fits_{self._fit_numbers[experiment]}"

    def _fit_number(self, experiment: str) -> int:
        """This session's fit number n for experiment (allocated on first save)."""
        self._fits_dir(experiment)
        return self._fit_numbers[experiment]

    def _next(self, _event):
        self._stash_state()
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
        # A file visited before comes back as it was left; a new one gets carried peaks.
        self._load_current(seed_peaks=seeds, state=self._states.get(self.current_idx))

    def _back(self, _event):
        if self.current_idx == 0:
            print("Already at the first file.")
            return
        self._stash_state()
        self.current_idx -= 1
        self._clear_fit()
        self._load_current(state=self._states.get(self.current_idx))


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def load_fit_results(csv_path: Path) -> pd.DataFrame:
    """Load fit_results.csv produced by PLFitter into a DataFrame."""
    return pd.read_csv(csv_path)


_SPECTRUM_KEYS = ['experiment', 'sample', 'spot', 'filename']   # filenames repeat across sweeps


def _latest_saves(df: pd.DataFrame, fit_number: int | None = None) -> pd.DataFrame:
    """
    Pick one fit per spectrum from fit_results.csv rows.

    The CSV holds one row set per (spectrum, fit session), tagged fit_number.
    With fit_number given, keep only that session's rows; otherwise keep each
    spectrum's highest fit_number.
    """
    data = df.copy()
    if fit_number is not None:
        return data[data['fit_number'] == fit_number]
    # Key columns can be blank (NaN) in the CSV; dropna=False keeps those groups.
    latest = data['fit_number'].groupby([data[k] for k in _SPECTRUM_KEYS],
                                        dropna=False).transform('max')
    return data[data['fit_number'] == latest]


def track_peak(df: pd.DataFrame, fit_number: int | None = None) -> pd.DataFrame:
    """
    Follow one peak through each field sweep: one row per fitted spectrum.

    For each (sample, spot), the starting peak is the largest-amplitude peak
    at the field closest to 0 (warns if no B = 0 spectrum). From there, walk
    outward through the fields in each direction, taking at each field the
    peak nearest in energy to the one tracked at the previous field. Spectra
    sharing a field are each matched to that previous energy. Each spectrum's
    latest fit (highest fit_number) is used, or the given fit_number only.
    """
    data = _latest_saves(df, fit_number)
    tracked = []
    for (sample, spot), sweep in data.groupby(['sample', 'spot'], dropna=False):
        sweep = sweep.dropna(subset=['field'])
        if sweep.empty:
            continue
        fields = np.sort(sweep['field'].unique())
        f0 = fields[np.argmin(np.abs(fields))]
        if f0 != 0:
            warnings.warn(f"{sample} / {spot}: no B = 0 spectrum; tracking from {f0} T.")
        at_f0 = sweep[sweep['field'] == f0]
        start = at_f0.loc[at_f0['amplitude'].idxmax()]

        for direction in (fields[fields > f0], fields[fields < f0][::-1], [f0]):
            ref = start['energy']
            for f in direction:
                chosen = []
                for _, spectrum in sweep[sweep['field'] == f].groupby('filename'):
                    chosen.append(spectrum.loc[(spectrum['energy'] - ref).abs().idxmin()])
                tracked.extend(chosen)
                ref = np.mean([row['energy'] for row in chosen])
    if not tracked:
        return data.iloc[0:0]
    return pd.DataFrame(tracked).sort_values(['sample', 'spot', 'field']).reset_index(drop=True)


def brightest_point(sweeps: list[Path], variable: str = 'raw',
                    emin: float | None = None, emax: float | None = None) -> pd.DataFrame:
    """
    Energy of the highest-intensity data point in each spectrum, vs. field.

    sweeps: sweep folders (raw CSV or processed .nc), one per sample/sweep.
    variable: 'raw' or 'subtracted' (processed .nc only). emin/emax optionally
    restrict the search to an energy window (eV). No fitting or smoothing:
    the result is the argmax of the data itself, so it is noise-limited and
    quantized to the energy grid.

    Returns one row per spectrum: sample, spot, field, energy, intensity —
    plottable with plot_peak_param(..., 'field', 'energy', group_by='sample').
    """
    rows = []
    for sweep in sweeps:
        bmap = _load_bfield_map(sweep, variable).sel(energy=slice(emin, emax))
        imax = bmap.argmax('energy')
        for field, energy, intensity in zip(bmap['field'].values,
                                            bmap['energy'][imax].values,
                                            bmap.isel(energy=imax).values):
            rows.append({'sample': bmap.attrs.get('sample', Path(sweep).name),
                         'spot': bmap.attrs.get('spot', ''),
                         'field': field, 'energy': energy, 'intensity': intensity})
    return pd.DataFrame(rows)


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