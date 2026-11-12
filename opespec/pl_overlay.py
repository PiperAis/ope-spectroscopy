"""
PL spectrum overlay: several steady-state spectra on one set of axes.

Key components:
- SpectrumOverlay: interactive window to compare any set of PL spectra
  (raw CSVs and/or processed .nc, from any folders). Spectra can be added
  (file dialog) and hidden/removed while it is open. Live controls: energy
  window, normalization, Savitzky–Golay smoothing, vertical offset, and
  raw ↔ baseline-subtracted data. Save writes the plot to a PNG.

Rules: display only — loaded data is never modified or written back. No
paths live here; callers pass the dialog start folder and the output folder.
Legend labels come from each spectrum's metadata attrs (filename /
metadata.txt via load_pl_data), not from the caller.

Loading and the shared GUI controls come from steady_state; smoothing from
pl_sweep. Unlike a sweep map, the spectra need not share an energy grid or
any metadata.
"""

import warnings
from datetime import datetime
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, CheckButtons
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from . import utilities
from .steady_state import load_pl_data, _spectrum_paths, _EnergyRangeControl, _CycleButton
from .pl_sweep import _savgol_along

xr.set_options(keep_attrs=True)


def _overlay_labels(spectra: list[dict]) -> list[str]:
    """
    Legend labels for loaded spectra: the metadata (sample, spot, field,
    temperature) that differs between them, e.g. 'flake1a, 0.2 T'. Falls back
    to the filename stem (then folder/stem) when that doesn't tell them apart.
    """
    formats = {'field': '{:g} T', 'temperature': '{:g} K'}
    keys = [k for k in ('sample', 'spot', 'field', 'temperature')
            if len({s['raw'].attrs.get(k) for s in spectra}) > 1]
    labels = []
    for s in spectra:
        parts = []
        for k in keys:
            value = s['raw'].attrs.get(k)
            if value is not None:
                parts.append(formats.get(k, '{}').format(value))
        labels.append(', '.join(parts))
    if keys and len(set(labels)) == len(labels) and all(labels):
        return labels
    stems = [s['path'].stem for s in spectra]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{s['path'].parent.name}/{s['path'].stem}" for s in spectra]


class SpectrumOverlay:
    """
    Interactive overlay of PL spectra (intensity vs. energy) on one plot.

    files: spectra to open with — a directory (its CSV / .nc files), a list
    of paths, or None to start empty. More can be added in the window.
    initial_dir: where the "Add files…" dialog opens. output_dir: where Save
    writes overlay_<YYYYmmdd-HHMMSS>.png (timestamped, so nothing is overwritten).

    Controls:
    - file checkboxes show/hide each spectrum; "Remove hidden" drops the
      unticked ones; "Add files…" opens a file dialog (.nc / .csv, any folder)
    - E min / E max (eV): the plotted window, clamped to the shown spectra
    - Normalize: Off → Global max (all shown spectra against the largest value
      in the window) → Per spectrum (each against its own max in the window)
    - Smooth (S-G) checkbox + window in points (smoothed before cropping)
    - Offset: spectrum i (counting shown ones) is shifted up by i × offset,
      in the plotted units (so ~1 suits normalized data)
    - Data: Raw ↔ Subtracted (baseline-subtracted, each spectrum's latest fit
      session). Subtracted needs every shown spectrum fitted; otherwise it
      falls back to Raw and says which aren't.

    Each spectrum keeps its colour when others are hidden or removed.

    Usage:
        SpectrumOverlay(paths, initial_dir=Path(cfg['processed_pl_dir']),
                        output_dir=Path(cfg['processed_pl_dir'])).run()
    """

    SMOOTH_POLYORDER = 3
    DEFAULT_SMOOTH_WINDOW = 11   # points along energy; must be odd and > polyorder
    _NORMS = ('off', 'global', 'spectrum')
    _NORM_NAMES = {'off': 'Off', 'global': 'Global max', 'spectrum': 'Per spectrum'}
    _SMOOTH_LABEL = 'Smooth (S-G)'
    _FILETYPES = [('PL spectra', '*.nc *.csv'), ('All files', '*.*')]

    def __init__(self, files: Path | list[Path] | None, initial_dir: Path, output_dir: Path):
        self.initial_dir = Path(initial_dir)
        self.output_dir = Path(output_dir)
        # One dict per loaded spectrum: path, raw, subtracted (None if unfitted),
        # shown, color. Order = load order = legend/offset order.
        self.spectra: list[dict] = []
        self._next_color = 0
        if files is not None:
            self._add(_spectrum_paths(files))

        self.variable = 'raw'
        self.normalize = 'off'
        self.smooth = False
        self.smooth_window = self.DEFAULT_SMOOTH_WINDOW
        self.offset = 0.0

        self.fig: Figure  # created in run()
        self.ax: Axes
        self.ax_files: Axes
        self.erange: _EnergyRangeControl
        self.cb_files: CheckButtons | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _add(self, paths: list[Path]) -> int:
        """Load paths not already loaded; return how many were added."""
        loaded = {s['path'].resolve() for s in self.spectra}
        added = 0
        for path in paths:
            path = Path(path)
            if path.resolve() in loaded:
                print(f"Already loaded: {path.name}")
                continue
            try:
                raw = load_pl_data(path, 'raw')
            except (KeyError, ValueError, OSError) as e:
                print(f"Could not load {path.name}: {e}")
                continue
            subtracted = None
            if path.suffix == '.nc':
                with warnings.catch_warnings():   # same metadata warnings as the raw load
                    warnings.simplefilter('ignore')
                    try:
                        subtracted = load_pl_data(path, 'subtracted')
                    except KeyError:
                        pass
            color = f"C{self._next_color % 10}"
            self._next_color += 1
            self.spectra.append({'path': path, 'raw': raw, 'subtracted': subtracted,
                                 'shown': True, 'color': color})
            loaded.add(path.resolve())
            added += 1
        return added

    @property
    def shown(self) -> list[dict]:
        return [s for s in self.spectra if s['shown']]

    # ------------------------------------------------------------------
    # Window
    # ------------------------------------------------------------------

    def run(self):
        self.fig = plt.figure(figsize=(12, 8), layout="constrained")
        outer = self.fig.add_gridspec(nrows=2, ncols=2, height_ratios=[4, 1],
                                      width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(outer[0, 0])

        side = outer[0, 1].subgridspec(nrows=2, ncols=1, height_ratios=[6, 1])
        self.ax_files = self.fig.add_subplot(side[0])
        buttons = side[1].subgridspec(nrows=2, ncols=1)
        self.btn_add = Button(self.fig.add_subplot(buttons[0]), 'Add files…')
        self.btn_remove = Button(self.fig.add_subplot(buttons[1]), 'Remove hidden')

        controls = outer[1, :].subgridspec(nrows=1, ncols=2, width_ratios=[3, 1])
        boxes = controls[0].subgridspec(nrows=1, ncols=4)
        right = controls[1].subgridspec(nrows=4, ncols=1)
        ax_emin, ax_emax, ax_win, ax_off = (self.fig.add_subplot(boxes[0, i]) for i in range(4))
        ax_emin.set_title('E min (eV)', fontsize=9)
        ax_emax.set_title('E max (eV)', fontsize=9)
        ax_win.set_title('Smooth window (pts)', fontsize=9)
        ax_off.set_title('Offset', fontsize=9)

        self.erange = _EnergyRangeControl(ax_emin, ax_emax, on_change=self._draw)
        self._update_bounds()
        self.tb_win = TextBox(ax_win, '', initial=str(self.smooth_window))
        self.tb_win.on_submit(self._on_window)
        self.tb_off = TextBox(ax_off, '', initial=f"{self.offset:g}")
        self.tb_off.on_submit(self._on_offset)

        self.cb_smooth = CheckButtons(self.fig.add_subplot(right[0]), [self._SMOOTH_LABEL],
                                      [self.smooth])
        self.cb_smooth.on_clicked(self._on_smooth)
        self.btn_norm = _CycleButton(self.fig.add_subplot(right[1]), 'Normalize', self._NORMS,
                                     self.normalize, on_change=self._on_normalize,
                                     display_names=self._NORM_NAMES)
        self.btn_data = _CycleButton(self.fig.add_subplot(right[2]), 'Data',
                                     ('raw', 'subtracted'), self.variable,
                                     on_change=self._on_data)
        self.btn_save = Button(self.fig.add_subplot(right[3]), 'Save')

        self.btn_add.on_clicked(self._on_add)
        self.btn_remove.on_clicked(self._on_remove)
        self.btn_save.on_clicked(self._save)

        self._build_file_list()
        self._draw()
        plt.show(block=True)

    def _build_file_list(self):
        """(Re)build the show/hide checkboxes for the loaded spectra."""
        if self.cb_files is not None:
            self.cb_files.disconnect_events()
            self.cb_files = None
        self.ax_files.cla()
        self.ax_files.set_title('Spectra', fontsize=9)
        if not self.spectra:
            self.ax_files.set_axis_off()
            self.ax_files.text(0.5, 0.5, 'None loaded', ha='center', va='center',
                               transform=self.ax_files.transAxes, fontsize=9)
            return
        self.ax_files.set_axis_on()
        self.cb_files = CheckButtons(
            self.ax_files, [s['path'].stem for s in self.spectra],
            [s['shown'] for s in self.spectra],
            label_props={'color': [s['color'] for s in self.spectra],
                         'fontsize': [8] * len(self.spectra)},
        )
        self.cb_files.on_clicked(self._on_toggle)

    # ------------------------------------------------------------------
    # Compute and draw
    # ------------------------------------------------------------------

    def _update_bounds(self):
        """E min/E max clamp limits = the union of the shown spectra's extents."""
        shown = [s[self.variable] for s in self.shown if s[self.variable] is not None]
        if not shown:
            return
        lo = min(float(d.energy.min()) for d in shown)
        hi = max(float(d.energy.max()) for d in shown)
        self.erange.set_data_bounds(xr.DataArray([0.0, 0.0], coords={'energy': [lo, hi]}))

    def _check_variable(self):
        """Fall back to raw if a shown spectrum has no baseline-subtracted data."""
        if self.variable != 'subtracted':
            return
        unfitted = [s['path'].name for s in self.shown if s['subtracted'] is None]
        if unfitted:
            print(f"No baseline-subtracted data (not fitted/saved, or a raw CSV), "
                  f"showing raw: {', '.join(unfitted)}")
            self.variable = 'raw'
            self.btn_data.set('raw')
            self._update_bounds()

    def _curves(self) -> list[tuple[dict, xr.DataArray]]:
        """(spectrum, processed curve) for each shown spectrum, before offset."""
        lo, hi = self.erange.range
        curves = []
        for s in self.shown:
            data = s[self.variable]
            if self.smooth:
                # Smooth the full spectrum before cropping, so the window edges
                # don't pick up filter edge effects.
                data = _savgol_along(data, 'energy', self.smooth_window, self.SMOOTH_POLYORDER)
            data = data.sel(energy=slice(lo, hi))
            if data.size == 0:
                continue
            if self.normalize == 'spectrum':
                data = data / float(data.max())
            curves.append((s, data))
        if self.normalize == 'global' and curves:
            peak = max(float(d.max()) for _, d in curves)
            curves = [(s, d / peak) for s, d in curves]
        return curves

    def _draw(self):
        self.ax.cla()
        if not self.shown:
            self.ax.text(0.5, 0.5, 'No spectra shown — tick one, or Add files…',
                         ha='center', va='center', transform=self.ax.transAxes)
            self.fig.canvas.draw_idle()
            return
        self._check_variable()
        labels = dict(zip((id(s) for s in self.spectra), _overlay_labels(self.spectra)))
        for i, (s, data) in enumerate(self._curves()):
            self.ax.plot(data.energy.values, data.values + i * self.offset,
                         color=s['color'], label=labels[id(s)])
        self.ax.set_xlim(*self.erange.range)
        self.ax.set_xlabel('Energy (eV)')
        ylabel = 'PL intensity'
        if self.variable == 'subtracted':
            ylabel += ', baseline-subtracted'
        ylabel += {'off': ' (arb. units)', 'global': ' (normalized)',
                   'spectrum': ' (normalized per spectrum)'}[self.normalize]
        if self.offset:
            ylabel += ', offset'
        self.ax.set_ylabel(ylabel)
        title = f'S-G smoothed, {self.smooth_window} pts' if self.smooth else ''
        self.ax.set_title(title, fontsize=9)
        self.ax.legend(fontsize=8)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_add(self, _event):
        paths = utilities.select_files(self.initial_dir, title="Add PL spectra to the overlay",
                                       filetypes=self._FILETYPES)
        if not paths:
            return
        if self._add(paths):
            self._build_file_list()
            self._update_bounds()
            self._draw()

    def _on_remove(self, _event):
        hidden = [s for s in self.spectra if not s['shown']]
        if not hidden:
            print("Nothing to remove: untick spectra first, then Remove hidden.")
            return
        self.spectra = [s for s in self.spectra if s['shown']]
        self._build_file_list()
        self._draw()

    def _on_toggle(self, label):
        assert self.cb_files is not None
        for s, state in zip(self.spectra, self.cb_files.get_status()):
            s['shown'] = state
        self._update_bounds()
        self._draw()

    def _on_data(self, variable: str):
        self.variable = variable
        self._check_variable()
        self._update_bounds()
        self._draw()

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

    def _on_offset(self, text):
        try:
            value = float(text)
        except ValueError:
            print(f"Invalid offset: {text!r}")
            self.tb_off.set_val(f"{self.offset:g}")
            return
        if value == self.offset:
            return
        self.offset = value
        self._draw()

    def _save(self, _event):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"overlay_{datetime.now():%Y%m%d-%H%M%S}.png"
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        bbox = self.ax.get_tightbbox(renderer).transformed(  # pyright: ignore[reportOptionalMemberAccess]
            self.fig.dpi_scale_trans.inverted())
        self.fig.savefig(out_path, dpi=200, bbox_inches=bbox)
        print(f"Saved -> {out_path}")
