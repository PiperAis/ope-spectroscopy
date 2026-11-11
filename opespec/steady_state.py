"""
Steady-state PL processing: loading, interactive fitting, and comparison.

Key components:
- load_pl_data: load a single PL CSV into an xr.DataArray (energy coords)
- PLFitter: interactive matplotlib fitter — click to place peaks, choose
  Gaussian or Lorentzian lineshape, optionally share (or lock) one width
  across chosen peaks and/or fit chosen peaks as an evenly spaced ladder,
  save results to CSV + PNG
- load_fit_results / plot_peak_param: load fit_results.csv and plot any
  parameter pair across datasets (energy vs field, width vs temperature, etc.)
- track_peak: follow one fitted peak through a field sweep

Sweep maps (spectra vs. field, contour builder) live in pl_sweep, which
builds on load_pl_data and _EnergyRangeControl from here.
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
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from . import fitting_functions as ff
from . import utilities

xr.set_options(keep_attrs=True)

# Metadata attrs every loaded spectrum carries (resolved by utilities.get_metadata
# from the filename + folder metadata.txt), and that a saved fit records.
_METADATA_KEYS = ('sample', 'spot', 'field', 'temperature')


def load_pl_data(filepath: Path, variable: str = 'raw') -> xr.DataArray:
    """
    Loads one PL spectrum into an xarray DataArray (energy coords, ascending).

    CSV (raw data): columns Wavelength, Intensity; wavelength converted to eV.
    Only variable='raw' is available from a CSV.

    .nc (processed, written by PLFitter on Save): a Dataset with variables
    'raw', 'baseline', 'subtracted'; `variable` picks one, and the Dataset's
    attrs (metadata + fit settings) are carried onto the returned array.
    A single-variable .nc (older format) is returned for variable='raw'.
    Metadata found in the filename or the folder's metadata.txt replaces the
    stored value (with a warning if they differ).
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
    # A value found there wins over one stored in a .nc at save time (warning
    # if they differ), so later metadata.txt edits take effect. PLFitter's Save
    # copies the raw folder's metadata.txt next to the .nc for this.
    # Stored attrs are kept when nothing is found; defaults fill the rest.
    found = utilities.get_metadata(filepath, defaults=False)
    for key in _METADATA_KEYS:
        value, stored = found.get(key), array.attrs.get(key)
        if value is not None:
            if stored is not None and stored != value:
                warnings.warn(f"{filepath.name}: {key}={value!r} from filename/"
                              f"{utilities.METADATA_FILENAME} replaces stored {stored!r}")
            array.attrs[key] = value
        elif stored is None:
            array.attrs[key] = utilities._METADATA_DEFAULTS[key]

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

    def set_data_bounds(self, data: xr.DataArray):
        """Set the clamp limits to data's energy extent. First call: range =
        full data range. Later calls: keep the current range, clamped to the
        new limits."""
        lo, hi = float(data.energy.min()), float(data.energy.max())
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


class _CycleButton:
    """
    Button that steps through a fixed list of options, one per click, labelled
    "<name>: <option> ▾". Shared by the interactive GUIs.

    Each click moves to the next option (wrapping), relabels, and calls
    on_change(value). set(value) changes it programmatically without calling
    on_change. Option display text is display_names[option] if given,
    else option.capitalize().
    """

    def __init__(self, ax: Axes, name: str, options, value: str, on_change,
                 display_names: dict | None = None):
        self.options = tuple(options)
        self.value = value
        self._name = name
        self._display = display_names or {}
        self._on_change = on_change
        self.button = Button(ax, self._label())
        self.button.on_clicked(self._on_click)

    def _label(self) -> str:
        return f"{self._name}: {self._display.get(self.value, self.value.capitalize())} ▾"

    def set(self, value: str):
        """Programmatically select value and relabel; no on_change."""
        self.value = value
        self.button.label.set_text(self._label())

    def _on_click(self, _event):
        i = self.options.index(self.value)
        self.set(self.options[(i + 1) % len(self.options)])
        self._on_change(self.value)
        self.button.ax.figure.canvas.draw_idle()


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


def _initial_baseline(x: np.ndarray, y: np.ndarray, baseline: str):
    """
    Initial guess and bounds for the baseline parameters, from the medians of
    the window's edge regions (robust to noise).

    Returns (params, bounds, xc): params is [c0] ('constant') or [c0, c1]
    ('linear') for b(x) = c0 [+ c1·(x − xc)], xc = window center.
    """
    xc = 0.5 * (float(x.min()) + float(x.max()))
    edge = max(1, len(y) // 10)
    xa, ya = float(np.mean(x[:edge])), float(np.median(y[:edge]))
    xb, yb = float(np.mean(x[-edge:])), float(np.median(y[-edge:]))
    if baseline == 'linear':
        c1 = (yb - ya) / (xb - xa) if xb != xa else 0.0
        c0 = ya + c1 * (xc - xa)
        return [c0, c1], [(float(y.min()), float(y.max())), (None, None)], xc
    c0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
    return [c0], [(float(y.min()), float(y.max()))], xc


def _peak_results(fit: np.ndarray, err: np.ndarray, cov: np.ndarray,
                  lineshape: str, x: np.ndarray) -> list[dict]:
    """
    Per-peak results from fitted peak parameters (amp, center, width per peak)
    and their 1σ errors / covariance: energy, width, amplitude, area (each with
    _err) and the peak's curve 'y' over x.
    """
    fn = ff.LINESHAPES[lineshape]
    peaks = []
    for j in range(len(fit) // 3):
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
    return peaks


def _constraint_map(p0: list, bounds: list, shared: list[bool], shared_width: float | None,
                    spaced: list[bool], center_bound: float = 0.008):
    """
    Map between the full peak-parameter vector p = [amp, center, width] × N
    (peaks in increasing energy) and the solver's free vector q, for the
    shared-width and equal-spacing constraints.

    Shared width: peaks with shared[j] use one common width — a free
    parameter, or the constant shared_width if given.
    Equal spacing: the r-th peak (in energy order) with spaced[j] has
    center E0 + r·Δ; E0 and Δ are free parameters. Needs ≥ 2 spaced peaks.
    E0 is bounded to ±center_bound around its initial value (the lowest
    spaced click), Δ to ±center_bound around the mean clicked spacing (and > 0).

    q = [per-peak free params in p order] + [shared width] + [E0, Δ]; the map
    is linear, p = M·q + c, so covariances map as M·cov_q·Mᵀ. With no
    constraints, q is p and M is the identity.

    Returns (q0, q_bounds, M, c, idx) with idx['spacing'] = position of Δ in q
    (None without a spacing group).
    """
    n = len(p0) // 3
    if sum(spaced) == 1:
        spaced = [False] * n      # a ladder needs two rungs
    cols, q0, qb = [], [], []     # cols[i]: {index into p: coefficient} for free param i
    c = np.zeros(len(p0))
    for j in range(n):
        own = [3*j] + ([] if spaced[j] else [3*j + 1]) + ([] if shared[j] else [3*j + 2])
        for i in own:
            cols.append({i: 1.0}); q0.append(p0[i]); qb.append(bounds[i])
    widths = [3*j + 2 for j in range(n) if shared[j]]
    if widths:
        if shared_width is None:
            cols.append({i: 1.0 for i in widths})
            q0.append(float(np.mean([p0[i] for i in widths])))
            qb.append(bounds[widths[0]])
        else:
            c[widths] = shared_width
    idx = {'spacing': None}
    centers = [3*j + 1 for j in range(n) if spaced[j]]
    if centers:
        e = [p0[i] for i in centers]
        e0, delta = e[0], (e[-1] - e[0]) / (len(e) - 1)
        cols.append({i: 1.0 for i in centers})
        q0.append(e0); qb.append([e0 - center_bound, e0 + center_bound])
        cols.append({i: float(r) for r, i in enumerate(centers)})
        q0.append(delta); qb.append([max(delta - center_bound, 1e-4), delta + center_bound])
        idx['spacing'] = len(cols) - 1
    M = np.zeros((len(p0), len(cols)))
    for k, col in enumerate(cols):
        for i, coef in col.items():
            M[i, k] = coef
    return q0, qb, M, c, idx


def _fit_spectrum(data: xr.DataArray, peak_positions: list,
                  lineshape: str, energy_range: tuple,
                  baseline: str = 'constant',
                  shared: list[bool] | None = None,
                  shared_width: float | None = None,
                  spaced: list[bool] | None = None) -> dict | None:
    """
    Fit N peaks to the spectrum within energy_range using clicked peak positions
    as initial guesses. Returns a result dict or None on failure.

    lineshape: 'gaussian' or 'lorentzian'
    energy_range: (xmin, xmax) in eV — the E min/E max range set in the GUI.
    baseline: 'constant' or 'linear' — fit jointly with the peaks as
              b(x) = c0 [+ c1·(x − xc)], xc = window center.
    shared: per peak (aligned with peak_positions), True = this peak's FWHM is
            the one shared width; default all False (every width free).
    shared_width: FWHM (eV) to lock the shared width at; None = fit it.
                  A locked width has zero fit uncertainty (width_err = 0).
    spaced: per peak, True = in the equal-spacing group: those peaks (in energy
            order) sit at E0, E0+Δ, E0+2Δ, … with E0 and Δ fitted. Needs ≥ 2
            peaks (a single one is ignored). Result has 'spacing'/'spacing_err'
            (eV; None without a group).
    With a spacing group, peak_positions must be in increasing energy order.
    """
    npk = len(peak_positions)
    shared = list(shared) if shared is not None else [False] * npk
    spaced = list(spaced) if spaced is not None else [False] * npk
    if len(shared) != npk or len(spaced) != npk:
        raise ValueError("shared/spaced must have one entry per peak position")
    if any(spaced) and list(peak_positions) != sorted(peak_positions):
        raise ValueError("with a spacing group, peak_positions must be in increasing energy order")
    if sum(spaced) == 1:
        print("Only one peak in the spacing group; fitting it without the spacing constraint.")
        spaced = [False] * npk
    xmin, xmax = energy_range
    mask = (data.energy >= xmin) & (data.energy <= xmax)
    da = data.where(mask, drop=True)

    if len(da) < 5:
        print("Too few data points in the E min/E max range.")
        return None

    x = da.energy.values.astype(float)
    y = da.values.astype(float)

    # Baseline is fit as extra parameter(s) alongside the peaks.
    bl0, bl_bounds, xc = _initial_baseline(x, y, baseline)
    k = len(bl0)
    bl_guess = _baseline_from_params(baseline, bl0, xc)

    # Initial amplitude: data value above the baseline guess at each clicked position.
    inputamps = []
    for pos in peak_positions:
        idx = int(np.argmin(np.abs(x - pos)))
        inputamps.append(max(float(y[idx] - _eval_baseline(bl_guess, x[idx])), 1.0))

    inputwids = [0.01] * len(peak_positions)
    pk0, pk_bounds = ff.generateParams_fixed(peak_positions, inputamps, inputwids)
    # Solver vector: free peak params (shared width once, or none if locked;
    # E0 and Δ for the spacing group), then baseline.
    q0, q_bounds, M, c, idx = _constraint_map(list(pk0), list(pk_bounds), shared,
                                              shared_width, spaced)
    p0 = q0 + bl0
    bounds = q_bounds + bl_bounds

    fn = ff.LINESHAPES[lineshape]

    # least_squares takes (lo, hi) arrays and needs lo < hi, so fixed parameters
    # can't be given equal bounds; a locked width is a constant in _constraint_map instead.
    lo = np.array([-np.inf if b[0] is None else b[0] for b in bounds], dtype=float)
    hi = np.array([np.inf if b[1] is None else b[1] for b in bounds], dtype=float)
    p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)

    def residuals(p):
        return (fn(M @ p[:-k] + c, x)
                + _eval_baseline(_baseline_from_params(baseline, p[-k:], xc), x) - y)

    try:
        # Residual-vector solver with Jacobian scaling: parameters span very different
        # magnitudes (heights ~1e3, centers ~1 eV, widths ~1e-2), which made the old
        # scalar L-BFGS-B fit stop well short of the minimum.
        result = least_squares(residuals, p0, bounds=(lo, hi), x_scale='jac')
    except Exception as e:
        # Only solver failures are reported this way; a bug in the result
        # processing below raises normally instead of looking like a failed fit.
        print(f"Fit error: {e}")
        return None

    bl = _baseline_from_params(baseline, result.x[-k:], xc)
    fit = M @ result.x[:-k] + c
    y_sub = y - _eval_baseline(bl, x)   # new name: `y` stays the raw data residuals() uses

    # 1σ uncertainties from the Jacobian (residual variance estimated from the fit),
    # mapped from the solver's free parameters back to every peak's (amp, center, width).
    cov_q = ff.param_covariance(result.jac, result.fun)[:-k, :-k]
    cov = M @ cov_q @ M.T
    err = np.sqrt(np.clip(np.diag(cov), 0, None))

    peaks = _peak_results(fit, err, cov, lineshape, x)
    for peak, s, sp in zip(peaks, shared, spaced):
        peak['shared'] = bool(s)
        peak['spaced'] = bool(sp)
    i_sp = idx['spacing']
    return {'peaks': peaks,
            'x_fit': x, 'y_total': fn(fit, x),
            'y_data': y_sub, 'baseline': bl,
            'width_locked': shared_width is not None and any(shared),
            'spacing': None if i_sp is None else float(result.x[i_sp]),
            'spacing_err': None if i_sp is None else float(np.sqrt(max(cov_q[i_sp, i_sp], 0.0))),
            **ff.fit_quality(result.fun, y, len(p0))}


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
    a row in df (e.g. a re-saved fit). Other rows are kept. If the existing file
    has different columns (an older schema), they are merged by name: df's
    columns first, extras after, and columns missing on either side become blank.
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


def _copy_folder_metadata(raw_folder: Path, processed_folder: Path):
    """
    Copy raw_folder's metadata.txt into processed_folder (if it exists and the
    copy differs), so a processed .nc resolves the same folder metadata as its
    raw file on load. The raw copy is the source of truth: re-saving
    overwrites edits made to the processed copy.
    """
    src = raw_folder / utilities.METADATA_FILENAME
    dst = processed_folder / utilities.METADATA_FILENAME
    if not src.exists() or (dst.exists() and dst.read_bytes() == src.read_bytes()):
        return
    if dst.exists():
        print(f"Updating {dst} from {src}")
    dst.write_bytes(src.read_bytes())
    utilities._read_folder_metadata.cache_clear()


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
    metadata = {k: raw.attrs.get(k) for k in ('filename', *_METADATA_KEYS)}
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
    # Shared-width constraint (netCDF has no bool attrs): 1 = in the shared group.
    attrs['peak_width_shared'] = np.array([int(p['shared']) for p in result['peaks']], dtype=np.int8)
    attrs['shared_width_locked'] = int(result['width_locked'])
    attrs['peak_energy_spaced'] = np.array([int(p['spaced']) for p in result['peaks']], dtype=np.int8)
    if result['spacing'] is not None:
        attrs['spacing'] = float(result['spacing'])
        attrs['spacing_err'] = float(result['spacing_err'])
    ds.attrs = attrs   # netCDF can't store None attrs; those were dropped above

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, mode='w')


def _fit_rows(filename: str, data: xr.DataArray, result: dict,
              experiment: str, fit_number: int) -> list[dict]:
    """fit_results.csv rows for one saved fit: one row per peak, with the
    spectrum's metadata, fit settings and fit quality repeated on each."""
    rows = []
    for j, peak in enumerate(result['peaks']):
        rows.append({
            'filename':    filename,
            'experiment':  experiment,
            'fit_number':  fit_number,
            **{k: data.attrs.get(k) for k in _METADATA_KEYS},
            'lineshape':   result['lineshape'],
            'baseline':    result['baseline']['type'],
            'fit_emin':    float(np.min(result['x_fit'])),
            'fit_emax':    float(np.max(result['x_fit'])),
            'peak_index':  j,
            'energy':        peak['energy'],
            'energy_err':    peak['energy_err'],
            'width':         peak['width'],
            'width_err':     peak['width_err'],
            'width_shared':  peak['shared'],
            'width_locked':  peak['shared'] and result['width_locked'],
            'energy_spaced': peak['spaced'],
            'spacing':       result['spacing'] if peak['spaced'] else None,
            'spacing_err':   result['spacing_err'] if peak['spaced'] else None,
            'amplitude':     peak['amplitude'],
            'amplitude_err': peak['amplitude_err'],
            'area':          peak['area'],
            'area_err':      peak['area_err'],
            'n_points':    result['n_points'],
            'dof':         result['dof'],
            'rmse':        result['rmse'],
            'r_squared':   result['r_squared'],
        })
    return rows


def _save_fit_plot(path: Path, result: dict, title: str):
    """Save a clean fit figure (no buttons or peak markers) to path: exactly the
    baseline-subtracted data that was fit, plus the fit overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result['x_fit'], result['y_data'], color='black', lw=1.2, label='Data')
    _plot_fit_overlay(ax, result)
    ax.set_title(title)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    Shift+left-click a marker to put it in or take it out of the
    shared-width group (magenta): those peaks are fitted with one common FWHM. After a
    fit the "Shared FWHM (meV)" box shows that width (or type one); with
    "Lock shared width" ticked, fits use the box value instead of fitting it
    (e.g. fit B = 0 unlocked, then lock for the rest of the sweep). The lock
    and value persist across files; Carry also carries each peak's group.
    Ctrl+left-click a marker to put it in or take it out of the equal-spacing
    group (thick dash-dot): those peaks, in energy order, are fitted as an even
    ladder E0, E0+Δ, E0+2Δ, … with E0 and Δ fitted (Δ ± error printed and
    saved). A peak can be in both groups; Carry carries this group too.
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
        # Shared-width constraint (persists across files): shift-click puts a marker
        # in the group; the box shows the last fitted shared FWHM, or a typed one,
        # and "Lock" fits with that value instead of fitting it.
        self.lock_width = False
        self.shared_width_mev: float | None = None

        self.fig: Figure  # created in run()
        self.ax: Axes
        self.erange: _EnergyRangeControl  # fit/display range; persists across files
        self._data_line: Line2D

    def run(self):
        if not self.files:
            print("No files to process.")
            return

        self.fig = plt.figure(figsize=(10, 7.5))
        self.ax  = self.fig.add_axes((0.10, 0.34, 0.87, 0.59))

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
        self.btn_base = _CycleButton(ax_base, 'Baseline', self._BASELINES, self.baseline,
                                     on_change=self._on_baseline)

        # Shared-width row, above the E range row.
        ax_sw = self.fig.add_axes((0.24, 0.205, 0.10, 0.05))
        self.tb_sw = TextBox(ax_sw, 'Shared FWHM (meV) ', initial='')
        self.tb_sw.on_submit(self._on_shared_width)
        ax_lock = self.fig.add_axes((0.36, 0.195, 0.24, 0.07))
        cb_lock = CheckButtons(ax_lock, ['Lock shared width'], [self.lock_width])
        cb_lock.on_clicked(self._on_lock)

        # Keep references so widgets aren't garbage-collected.
        self._buttons = [btn_fitg, btn_fitl, btn_clear, btn_save, btn_back, btn_next, cb_carry,
                         *self.erange.widgets, self.btn_base, self.tb_sw, cb_lock]

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        self._load_current()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: file loading
    # ------------------------------------------------------------------

    def _load_current(self, seed_peaks: list[tuple[float, bool, bool]] | None = None,
                      state: dict | None = None):
        """Load files[current_idx]. `state` (a snapshot from _stash_state) restores a
        previously visited file; otherwise `seed_peaks` (carried (x, shared, spaced)
        peaks) are placed."""
        self.peak_markers.clear()
        self.fit_lines.clear()
        self.fit_result = None
        self._saved = False

        filepath = self.files[self.current_idx]
        self.data = load_pl_data(filepath)

        # First file: full data range. Later files: keep the previous range,
        # clamped in case this file covers less energy.
        self.erange.set_data_bounds(self.data)
        if state is not None:
            self.erange.set_range(*state['range'])
            self.baseline = state['baseline']
            self.btn_base.set(self.baseline)

        self.ax.cla()
        cropped = self._current_data()
        self._data_line, = self.ax.plot(cropped.energy.values, cropped.values,
                                        color='black', lw=1.2, label='Data')
        self._set_view()
        self.ax.set_xlabel('Energy (eV)')
        self.ax.set_ylabel('Intensity (arb. units)')

        markers = state['markers'] if state is not None else (seed_peaks or [])
        for x, shared, spaced in markers:
            self._add_peak(x, shared, spaced)
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
            'left-click: place  |  right-click: remove  |  shift-click: shared width  |  '
            'ctrl-click: equal spacing',
            fontsize=9
        )

    def _stash_state(self):
        """Snapshot the current file so Back / revisiting restores it."""
        self._states[self.current_idx] = {
            'markers':    [(m['x'], m['shared'], m['spaced']) for m in self.peak_markers],
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

    def _on_lock(self, _label):
        self.lock_width = not self.lock_width

    def _on_shared_width(self, text: str):
        if not text.strip():
            self.shared_width_mev = None
            return
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            print(f"Invalid shared FWHM: {text!r} (meV, > 0)")
            self.tb_sw.set_val('' if self.shared_width_mev is None else f'{self.shared_width_mev:.4g}')
            return
        self.shared_width_mev = value

    _BASELINES = ('constant', 'linear')

    def _on_baseline(self, baseline: str):
        self.baseline = baseline
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
        if event.button == 1 and event.key == 'shift':
            self._toggle_shared(event.xdata)
        elif event.button == 1 and event.key == 'control':
            self._toggle_spaced(event.xdata)
        elif event.button == 1:
            self._add_peak(event.xdata)
        elif event.button == 3:
            self._remove_nearest_peak(event.xdata)

    _MARKER_COLORS = {False: 'steelblue', True: 'magenta'}   # shared width; not a tab10 fit colour

    _MARKER_STYLES = {False: '--', True: '-.'}   # equal-spacing group → dash-dot

    def _add_peak(self, x, shared: bool = False, spaced: bool = False):
        color = self._MARKER_COLORS[shared]
        line = self.ax.axvline(x, color=color, linestyle=self._MARKER_STYLES[spaced],
                               alpha=0.8, lw=1.8 if spaced else 1.2)
        # Blended transform: data x, axes-fraction y — stays at top regardless of zoom.
        text = self.ax.text(
            x, 0.96, f'{x:.4f}',
            transform=self.ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=8, color=color, rotation=90,
        )
        self.peak_markers.append({'x': x, 'line': line, 'text': text,
                                  'shared': shared, 'spaced': spaced})
        self.fig.canvas.draw_idle()

    def _toggle_shared(self, x):
        """Move the marker nearest x into / out of the shared-width group."""
        if not self.peak_markers:
            return
        m = min(self.peak_markers, key=lambda m: abs(m['x'] - x))
        m['shared'] = not m['shared']
        m['line'].set_color(self._MARKER_COLORS[m['shared']])
        m['text'].set_color(self._MARKER_COLORS[m['shared']])
        self.fig.canvas.draw_idle()

    def _toggle_spaced(self, x):
        """Move the marker nearest x into / out of the equal-spacing group."""
        if not self.peak_markers:
            return
        m = min(self.peak_markers, key=lambda m: abs(m['x'] - x))
        m['spaced'] = not m['spaced']
        m['line'].set_linestyle(self._MARKER_STYLES[m['spaced']])
        m['line'].set_linewidth(1.8 if m['spaced'] else 1.2)
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

        markers = sorted(self.peak_markers, key=lambda m: m['x'])
        peak_positions = [m['x'] for m in markers]
        shared = [m['shared'] for m in markers]
        spaced = [m['spaced'] for m in markers]
        shared_width = None
        if self.lock_width and any(shared):
            if self.shared_width_mev is None:
                print("Lock shared width is on, but no shared FWHM is set: fit once "
                      "unlocked, or type a value (meV).")
                return
            shared_width = self.shared_width_mev / 1e3
        result = _fit_spectrum(self.data, peak_positions, lineshape, self.erange.range,
                               baseline=self.baseline, shared=shared, shared_width=shared_width,
                               spaced=spaced)
        if result is None:
            print("Fit failed — try adjusting peak positions or the E min/E max range.")
            return

        self.fit_result = {**result, 'lineshape': lineshape}
        self._saved = False
        self._show_fit()
        if any(shared) and shared_width is None:
            # Show the fitted shared width; with Lock, later fits (and files) reuse it.
            self.tb_sw.set_val(f"{result['peaks'][shared.index(True)]['width'] * 1e3:.4g}")

        print(f"\nFit ({lineshape}): RMSE={result['rmse']:.3g}, R²={result['r_squared']:.4f}")
        if result['spacing'] is not None:
            print(f"  Spacing Δ = {result['spacing'] * 1e3:.3f} ± {result['spacing_err'] * 1e3:.2g} meV")
        for j, peak in enumerate(result['peaks']):
            tags = []
            if peak['shared']:
                tags.append('shared width, locked' if result['width_locked'] else 'shared width')
            if peak['spaced']:
                tags.append('equal spacing')
            tag = f"  [{'; '.join(tags)}]" if tags else ''
            print(f"  Peak {j+1}: E={peak['energy']:.4f} ± {peak['energy_err']:.2g} eV | "
                  f"FWHM={peak['width']:.4f} ± {peak['width_err']:.2g} eV | "
                  f"amp={peak['amplitude']:.1f} ± {peak['amplitude_err']:.2g} | "
                  f"area={peak['area']:.4f} ± {peak['area_err']:.2g}{tag}")

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
        result = self.fit_result
        experiment = filepath.parent.name
        fit_number = self._fit_number(experiment)

        # One row set per (spectrum, fit session): re-saving in this session replaces it.
        rows = _fit_rows(filepath.name, self.data, result, experiment, fit_number)
        csv_path = self.output_dir / 'fit_results.csv'
        _upsert_rows_csv(csv_path, pd.DataFrame(rows),
                         keys=['experiment', 'filename', 'fit_number'])
        print(f"Saved {len(rows)} peak(s) as fit {fit_number} → {csv_path}")

        plot_path = self._fits_dir(experiment) / (filepath.stem + '_fit.png')
        _save_fit_plot(plot_path, result, title=f"{filepath.name} fit "
                                                f"({result['baseline']['type']} baseline)")
        print(f"Plot → {plot_path}")

        # Processed spectrum (raw / baseline / subtracted) for the contour builder,
        # one .nc per spectrum under <output_dir>/<sweep folder>/, like TRR's per-set .nc.
        nc_path = self.output_dir / experiment / f"{filepath.stem}.nc"
        _save_processed_spectrum(nc_path, self.data, result,
                                 result['lineshape'], experiment, fit_number)
        print(f"Spectrum → {nc_path}")
        _copy_folder_metadata(filepath.parent, nc_path.parent)

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
                seeds = [(p['energy'], p['shared'], p['spaced']) for p in self.fit_result['peaks']]
            else:
                seeds = [(m['x'], m['shared'], m['spaced']) for m in self.peak_markers]

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