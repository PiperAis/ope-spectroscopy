"""
PL sweeps: a set of steady-state spectra taken while one variable (magnetic
field or temperature) is stepped, combined into one 2D (sweep, energy) map.

Key components:
- SWEEPS: the supported sweep variables (attr name → axis label, units,
  output-filename tag)
- _load_sweep_map: load a sweep folder (raw CSVs or processed .nc) into a
  2D DataArray sorted by the sweep variable
- SweepContourBuilder: interactive heatmap builder for PL intensity vs
  energy and the sweep variable — colormap, energy bounds, and normalization
  live, save to PNG when done
- brightest_point: energy of each spectrum's highest data point vs. the sweep variable

Rules: the sweep variable is always passed explicitly (sweep='field' by
default) — never inferred from the data, since a folder can vary in more
than one. Each spectrum's value comes from its attrs (filename /
metadata.txt via utilities.get_metadata). No paths live here; callers pass them.

Single-spectrum loading, fitting and fit-result comparison live in
steady_state; this module builds on its loader and GUI helpers.
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as smooth
from matplotlib.widgets import Button, TextBox, CheckButtons
from matplotlib.transforms import Bbox
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from . import utilities
from .steady_state import (load_pl_data, _spectrum_paths, _EnergyRangeControl, _CycleButton,
                           _METADATA_KEYS)

xr.set_options(keep_attrs=True)

# Sweep variable (a spectrum attr from utilities.get_metadata) → axis label,
# units, and the tag used in saved contour filenames.
SWEEPS = {
    'field':       {'long_name': 'Magnetic Field', 'units': 'T', 'tag': 'bfield'},
    'temperature': {'long_name': 'Temperature',    'units': 'K', 'tag': 'temperature'},
}


def _check_sweep(sweep: str) -> dict:
    if sweep not in SWEEPS:
        raise ValueError(f"Unknown sweep variable {sweep!r}; options: {list(SWEEPS)}")
    return SWEEPS[sweep]


def _has_variable(filepath: Path, variable: str) -> bool:
    """True if filepath is a processed .nc containing `variable`."""
    if filepath.suffix != '.nc':
        return variable == 'raw'
    with xr.open_dataset(filepath) as ds:
        return variable in ds.data_vars


def _load_sweep_map(files: Path | list[Path], variable: str = 'raw',
                    sweep: str = 'field') -> xr.DataArray:
    """
    Loads a set of PL spectra into one 2D (sweep, energy) DataArray.

    files: a directory (its CSV / processed .nc files are used) or an explicit
    list, one spectrum per sweep value. variable: 'raw', or 'subtracted' /
    'baseline' from processed .nc files. sweep: a key of SWEEPS ('field' or
    'temperature'); each spectrum's value comes from its attrs (filename /
    metadata.txt). Sorted by sweep value (not acquisition order), since
    B-field sweeps commonly ramp up then back down through zero.

    The sweep attr becomes the coordinate. The other metadata attrs must
    agree across the spectra (missing/None values don't count as differing):
    - physical parameters (the other SWEEPS variable, e.g. temperature in a
      field sweep): ValueError, since a folder mixing them is not one sweep
      (pass an explicit file list to load a consistent subset);
    - labels (sample, spot): warning, and the label is dropped from the map's
      attrs rather than taking the first spectrum's value.
    For processed spectra, attrs fit_emin/fit_emax on the map are the
    intersection of the individual fit windows.
    """
    info = _check_sweep(sweep)
    spectra, windows = [], []
    others = [k for k in _METADATA_KEYS if k != sweep]
    seen = {k: {} for k in others}   # attr → {value: first filename with it}
    for filepath in _spectrum_paths(files):
        da = load_pl_data(filepath, variable=variable)
        value = da.attrs[sweep]
        if value is None:
            warnings.warn(f"Could not parse {info['long_name'].lower()} from filename or "
                          f"{utilities.METADATA_FILENAME}: {filepath.name}")
        for k in others:
            if da.attrs.get(k) is not None:
                seen[k].setdefault(da.attrs[k], filepath.name)
        if 'fit_emin' in da.attrs:
            windows.append((da.attrs['fit_emin'], da.attrs['fit_emax']))
        spectra.append(da.expand_dims({sweep: [value]}))

    def describe(mixed):
        return '; '.join(f"{k}: " + ', '.join(f"{val!r} (e.g. {name})" for val, name in v.items())
                         for k, v in mixed.items())
    mixed = {k: v for k, v in seen.items() if len(v) > 1}
    physical = {k: v for k, v in mixed.items() if k in SWEEPS}
    labels = {k: v for k, v in mixed.items() if k not in SWEEPS}
    if physical:
        raise ValueError(f"Spectra in this {sweep} sweep differ in {describe(physical)}. "
                         f"Pass an explicit file list to load a consistent subset.")
    if labels:
        warnings.warn(f"Spectra in this {sweep} sweep have inconsistent labels — "
                      f"{describe(labels)}. Dropping {', '.join(labels)} from the map.")

    sweep_map = xr.concat(spectra, dim=sweep).sortby(sweep)
    for k in labels:
        sweep_map.attrs.pop(k, None)
    # Drop per-spectrum attrs (now a coordinate, or specific to one fit).
    for key in list(sweep_map.attrs):
        if key in (sweep, 'filename', 'lineshape', 'fit_emin', 'fit_emax',
                   'fit_number', 'rmse', 'r_squared') \
                or key.startswith(('peak_', 'baseline_')):
            del sweep_map.attrs[key]
    if windows and len(windows) == len(spectra):
        sweep_map.attrs['fit_emin'] = max(w[0] for w in windows)
        sweep_map.attrs['fit_emax'] = min(w[1] for w in windows)
    sweep_map[sweep].attrs = {'long_name': info['long_name'], 'units': info['units']}
    sweep_map['energy'].attrs['long_name'] = 'Energy'
    return sweep_map


def _savgol_along(da: xr.DataArray, dim: str, window: int, polyorder: int) -> xr.DataArray:
    """Savitzky–Golay smooth `da` along `dim` (window in points; shrunk to fit if too long)."""
    n = da.sizes[dim]
    window = min(window, n if n % 2 else n - 1)
    if window <= polyorder:
        return da
    return xr.apply_ufunc(lambda a: smooth(a, window, polyorder, axis=-1), da,
                          input_core_dims=[[dim]], output_core_dims=[[dim]], keep_attrs=True)


class SweepContourBuilder:
    """
    Interactive contour (heatmap) builder for PL intensity vs. energy and a
    sweep variable: sweep='field' (default) or 'temperature' (see SWEEPS).

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
        SweepContourBuilder(files, output_dir=Path(cfg['processed_pl_dir'])).run()
        SweepContourBuilder(files, output_dir=..., sweep='temperature').run()

    Save writes <output_dir>/<sample>_<tag>_contour[_subtracted][_sg<N>].png,
    with <tag> from SWEEPS (e.g. 'bfield').
    """

    DEFAULT_CMAP = 'inferno'
    _VARIABLES = ('subtracted', 'raw')
    SMOOTH_POLYORDER = 3
    DEFAULT_SMOOTH_WINDOW = 11   # points along energy; must be odd and > polyorder

    def __init__(self, files: Path | list[Path], output_dir: Path, sweep: str = 'field'):
        _check_sweep(sweep)
        self.sweep = sweep
        paths = _spectrum_paths(files)
        self.maps = {'raw': _load_sweep_map(paths, 'raw', sweep)}
        unfitted = [p.name for p in paths if not _has_variable(p, 'subtracted')]
        if not unfitted:
            self.maps['subtracted'] = _load_sweep_map(paths, 'subtracted', sweep)
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
        self.erange.set_data_bounds(self.data)
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
        self.btn_norm = _CycleButton(ax_norm, 'Normalize', self._NORMS, self.normalize,
                                     on_change=self._on_normalize, display_names=self._NORM_NAMES)
        self.btn_save = Button(ax_save, 'Save')
        if has_toggle:
            self.btn_data = _CycleButton(self.fig.add_subplot(side_gs[2, 0]), 'Data',
                                         [v for v in self._VARIABLES if v in self.maps],
                                         self.variable, on_change=self._on_data)

        self.tb_cmap.on_submit(self._on_cmap)
        self.cb_smooth.on_clicked(self._on_smooth)
        self.btn_save.on_clicked(self._save)

        self._draw()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: compute and draw
    # ------------------------------------------------------------------

    @property
    def data(self) -> xr.DataArray:
        """The (sweep, energy) map currently shown: raw or baseline-subtracted."""
        return self.maps[self.variable]

    def _on_data(self, variable: str):
        # Switch raw ↔ subtracted; keep the E range (clamped to the new map).
        self.variable = variable
        self.erange.set_data_bounds(self.data)
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
        result = data.plot.contourf(x=self.sweep, y='energy', cmap=self.cmap,
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

    def _on_normalize(self, mode: str):
        self.normalize = mode
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
        tag = SWEEPS[self.sweep]['tag']
        sample = self.data.attrs.get('sample', tag)
        suffix = '_subtracted' if self.variable == 'subtracted' else ''
        if self.smooth:
            suffix += f'_sg{self.smooth_window}'
        out_path = self.output_dir / f"{sample.replace(' ', '')}_{tag}_contour{suffix}.png"

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        assert self._cbar is not None
        bbox = Bbox.union([  # pyright: ignore[reportArgumentType]
            self.ax.get_tightbbox(renderer),
            self._cbar.ax.get_tightbbox(renderer),
        ]).transformed(self.fig.dpi_scale_trans.inverted())

        self.fig.savefig(out_path, dpi=200, bbox_inches=bbox)
        print(f"Saved -> {out_path}")


def brightest_point(folders: list[Path], variable: str = 'raw',
                    emin: float | None = None, emax: float | None = None,
                    sweep: str = 'field') -> pd.DataFrame:
    """
    Energy of the highest-intensity data point in each spectrum, vs. the
    sweep variable (sweep='field' or 'temperature').

    folders: sweep folders (raw CSV or processed .nc), one per sample/sweep.
    variable: 'raw' or 'subtracted' (processed .nc only). emin/emax optionally
    restrict the search to an energy window (eV). No fitting or smoothing:
    the result is the argmax of the data itself, so it is noise-limited and
    quantized to the energy grid.

    Returns one row per spectrum: sample, spot, <sweep>, energy, intensity —
    plottable with plot_peak_param(..., 'field', 'energy', group_by='sample').
    """
    rows = []
    for folder in folders:
        smap = _load_sweep_map(folder, variable, sweep).sel(energy=slice(emin, emax))
        imax = smap.argmax('energy')
        for value, energy, intensity in zip(smap[sweep].values,
                                            smap['energy'][imax].values,
                                            smap.isel(energy=imax).values):
            rows.append({'sample': smap.attrs.get('sample', Path(folder).name),
                         'spot': smap.attrs.get('spot', ''),
                         sweep: value, 'energy': energy, 'intensity': intensity})
    return pd.DataFrame(rows)
