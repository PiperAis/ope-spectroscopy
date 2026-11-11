"""
Cross-dataset comparison of saved PL fits (fit_results.csv from PLFitter):
follow fitted peaks through a field sweep and fit their energy trend.

Key components:
- track_branches: follow every peak of the B ≈ 0 fit through each sweep
  (experiment), nearest energy field to field; flags points shared by
  more than one branch (peaks merged at high field)
- fit_branch: weighted polynomial E(B) = E0 + a·B (+ b·B²) for one branch
- FitCompareViewer: interactive window — tracked branches (energy vs field)
  per experiment with error bars, shared/bad-fit flags and trend fits;
  branch toggles, fit-session picker, save PNG + coefficient CSV;
  click a point to view that spectrum's saved fit plot

Rules: works only on the fit_results DataFrame (load_fit_results) — no
refitting. A "series" is an experiment (sweep folder). Each spectrum's
latest fit session is used unless fit_number is given. No paths live here;
callers pass output_dir (the processed_pl_dir PLFitter saved to).

Loading, fitting and the single-peak track_peak live in steady_state.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button, CheckButtons
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .steady_state import _latest_saves, _CycleButton


def track_branches(df: pd.DataFrame, fit_number: int | None = None) -> pd.DataFrame:
    """
    Follow every peak of the B ≈ 0 fit through each field sweep.

    Per experiment, the seeds are all peaks of the spectrum at the field
    closest to 0 (warns if that field isn't 0; if several spectra share it,
    the first by filename seeds). Each seed is walked outward through the
    fields in both directions, taking at each field the peak nearest in
    energy to the one tracked at the previous field (as track_peak does for
    the brightest peak only).

    Returns one row per (branch, spectrum) with added columns:
      branch     0, 1, ... in order of seed energy, per experiment
      branch_e0  the seed energy (eV)
      shared     True where this fitted peak is also on another branch
    """
    data = _latest_saves(df, fit_number)
    tracked = []
    for experiment, sweep in data.groupby('experiment', dropna=False):
        sweep = sweep.dropna(subset=['field'])
        if sweep.empty:
            continue
        fields = np.sort(sweep['field'].unique())
        f0 = fields[np.argmin(np.abs(fields))]
        if f0 != 0:
            warnings.warn(f"{experiment}: no B = 0 spectrum; seeding branches at {f0} T.")
        at_f0 = sweep[sweep['field'] == f0]
        seed_file = sorted(at_f0['filename'])[0]
        seeds = np.sort(at_f0[at_f0['filename'] == seed_file]['energy'].to_numpy())

        for branch, e0 in enumerate(seeds):
            for direction in (fields[fields > f0], fields[fields < f0][::-1], [f0]):
                ref = e0
                for f in direction:
                    chosen = []
                    for _, spectrum in sweep[sweep['field'] == f].groupby('filename'):
                        row = spectrum.loc[(spectrum['energy'] - ref).abs().idxmin()].copy()
                        row['branch'] = branch
                        row['branch_e0'] = e0
                        chosen.append(row)
                    tracked.extend(chosen)
                    ref = np.mean([row['energy'] for row in chosen])
    if not tracked:
        return data.iloc[0:0].assign(branch=[], branch_e0=[], shared=[])

    out = pd.DataFrame(tracked)
    peak_id = ['experiment', 'filename', 'peak_index']
    n_branches = out.groupby(peak_id, dropna=False)['branch'].transform('nunique')
    out['shared'] = n_branches > 1
    return out.sort_values(['experiment', 'branch', 'field']).reset_index(drop=True)


def fit_branch(rows: pd.DataFrame, degree: int = 2,
               include_shared: bool = False) -> dict | None:
    """
    Fit one branch's energy vs field: E(B) = E0 + a·B (+ b·B² for degree 2).

    rows : one branch from track_branches (one experiment, one branch)
    include_shared : if False (default), points shared with another branch
                     (merged peaks) are left out

    Weighted by 1/energy_err; unweighted if any point lacks an error (older
    fit_results rows). Coefficient errors are scaled by the reduced χ², so
    they reflect the actual scatter about the trend. Returns None if there
    are too few points (needs more than degree + 1).

    Returns {'E0', 'a', ['b'], 'E0_err', 'a_err', ['b_err'], 'n_points', 'degree'}.
    """
    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    pts = rows if include_shared else rows[~rows['shared'].astype(bool)]
    pts = pts.dropna(subset=['field', 'energy'])
    n = len(pts)
    if n <= degree + 1:
        return None

    err = pts['energy_err'].to_numpy(dtype=float) if 'energy_err' in pts else np.full(n, np.nan)
    w = 1 / err if np.all(np.isfinite(err) & (err > 0)) else None
    coeffs, cov = np.polyfit(pts['field'].to_numpy(dtype=float), pts['energy'].to_numpy(dtype=float),
                             degree, w=w, cov=True)
    errs = np.sqrt(np.diag(cov))

    # polyfit returns highest power first.
    names = ['E0', 'a', 'b'][:degree + 1]
    result = {'n_points': n, 'degree': degree}
    for name, c, e in zip(names, coeffs[::-1], errs[::-1]):
        result[name] = float(c)
        result[f'{name}_err'] = float(e)
    return result


class FitCompareViewer:
    """
    Interactive window: fitted peak energy vs field, followed as branches
    (track_branches) through each experiment, with a trend fit per branch.

    - Grey points: every fitted peak in the selected session (context).
    - Branches: colour = experiment, marker = branch, error bars = energy_err.
      Hollow markers are shared with another branch (merged peaks); a black ×
      flags a doubtful fit (r_squared < R2_MIN, or energy_err/width > REL_ERR_MAX).
    - Dashed line: fit_branch trend (quadratic); coefficients listed below.

    Controls: one checkbox per (experiment, branch); "Session" cycles
    latest → each fit_number; "Include shared points in trend fit"; Save.
    Clicking any point (branch or grey) shows that spectrum's saved fit plot,
    <output_dir>/<experiment>/plots/fits_<fit_number>/<stem>_fit.png, in a
    separate window that is reused by later clicks.

    Save writes <output_dir>/branch_compare_<session>.png and
    <output_dir>/branch_fits_<session>.csv (coefficients of the visible
    branches; energies in eV, a in eV/T, b in eV/T²). Both are overwritten
    by a later save of the same session.

    Must be run as a plain script (like PLFitter), not in a #%% cell.

    Usage:
        df = load_fit_results(Path(cfg['processed_pl_dir']) / 'fit_results.csv')
        FitCompareViewer(df, output_dir=Path(cfg['processed_pl_dir'])).run()
    """

    R2_MIN = 0.95
    REL_ERR_MAX = 0.5   # energy_err / width: centre not resolved within the line
    DEGREE = 2
    _MARKERS = 'osD^v<>ph*'

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = Path(output_dir)
        sessions = sorted(int(n) for n in df['fit_number'].dropna().unique())
        self._sessions = ['latest'] + [str(n) for n in sessions]
        self.session = 'latest'
        self.include_shared = False
        self.visible: dict[tuple, bool] = {}   # (experiment, branch) → shown; persists across sessions
        self.experiments = sorted(df['experiment'].dropna().astype(str).unique())

        self.fig: Figure  # created in run()
        self.ax: Axes
        self.ax_toggles: Axes
        self.ax_text: Axes
        self.cb_branches: CheckButtons | None = None
        self.branches = pd.DataFrame()
        self.fits: dict[tuple, dict | None] = {}
        self._pick_rows: dict = {}             # plotted point artist → its rows, for click-to-open
        self._last_click = None                # one click can pick several overlapping artists
        self._preview: Figure | None = None

    @property
    def fit_number(self) -> int | None:
        return None if self.session == 'latest' else int(self.session)

    def run(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes((0.08, 0.42, 0.60, 0.53))
        self.ax_toggles = self.fig.add_axes((0.72, 0.42, 0.27, 0.53))
        self.ax_text = self.fig.add_axes((0.02, 0.02, 0.96, 0.20))
        self.ax_text.axis('off')

        ax_session = self.fig.add_axes((0.08, 0.25, 0.18, 0.05))
        ax_shared = self.fig.add_axes((0.28, 0.24, 0.26, 0.07))
        ax_save = self.fig.add_axes((0.56, 0.25, 0.12, 0.05))
        self.btn_session = _CycleButton(ax_session, 'Session', self._sessions, self.session,
                                        on_change=self._on_session)
        self.cb_shared = CheckButtons(ax_shared, ['Include shared points\nin trend fit'],
                                      [self.include_shared])
        self.cb_shared.on_clicked(self._on_shared)
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._save)

        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self._load_session()
        plt.show(block=True)

    # ------------------------------------------------------------------
    # Internal: data
    # ------------------------------------------------------------------

    def _load_session(self):
        """Re-track branches for the selected session and rebuild the toggles."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.branches = track_branches(self.df, self.fit_number)
        for w in caught:
            print(f"Warning: {w.message}")
        self._fit_all()
        self._build_toggles()
        self._draw()

    def _fit_all(self):
        self.fits = {}
        for key, rows in self.branches.groupby(['experiment', 'branch']):
            self.fits[key] = fit_branch(rows, self.DEGREE, self.include_shared)

    def _keys(self) -> list[tuple]:
        return list(self.fits)

    def _rows(self, key: tuple) -> pd.DataFrame:
        exp, br = key
        return self.branches[(self.branches['experiment'] == exp)
                             & (self.branches['branch'] == br)].sort_values('field')

    def _label(self, key: tuple) -> str:
        return f"{key[0]} · {self._rows(key)['branch_e0'].iloc[0]:.4f} eV"

    def _color(self, experiment) -> tuple:
        colors = plt.cm.tab10.colors  # pyright: ignore[reportAttributeAccessIssue]
        i = self.experiments.index(str(experiment)) if str(experiment) in self.experiments else 0
        return colors[i % 10]

    def _bad(self, rows: pd.DataFrame) -> pd.Series:
        rel = rows['energy_err'] / rows['width']
        return (rows['r_squared'] < self.R2_MIN) | (rel > self.REL_ERR_MAX)

    # ------------------------------------------------------------------
    # Internal: drawing
    # ------------------------------------------------------------------

    def _build_toggles(self):
        self.ax_toggles.cla()
        keys = self._keys()
        for k in keys:
            self.visible.setdefault(k, True)
        if not keys:
            self.ax_toggles.axis('off')
            self.cb_branches = None
            return
        self.ax_toggles.axis('on')
        self.cb_branches = CheckButtons(self.ax_toggles, [self._label(k) for k in keys],
                                        [self.visible[k] for k in keys])
        for text in self.cb_branches.labels:
            text.set_fontsize(8)
        self.cb_branches.on_clicked(self._on_toggle)
        self.ax_toggles.set_title('Branches (experiment · E at B≈0)', fontsize=9)

    def _draw(self):
        ax = self.ax
        ax.cla()
        self._pick_rows = {}
        context = _latest_saves(self.df, self.fit_number).dropna(subset=['field'])
        artist, = ax.plot(context['field'], context['energy'], '.', color='0.8', zorder=1,
                          label='All fitted peaks', picker=5)
        self._pick_rows[artist] = context

        lines = []
        for key, fit in self.fits.items():
            if not self.visible.get(key, True):
                continue
            rows = self._rows(key)
            color = self._color(key[0])
            marker = self._MARKERS[int(key[1]) % len(self._MARKERS)]
            shared = rows['shared'].astype(bool)
            ax.errorbar(rows['field'], rows['energy'], yerr=rows['energy_err'],
                        color=color, lw=0.8, capsize=2, marker='none', zorder=2)
            artist, = ax.plot(rows['field'][~shared], rows['energy'][~shared], marker,
                              color=color, zorder=3, label=self._label(key), picker=5)
            self._pick_rows[artist] = rows[~shared]
            artist, = ax.plot(rows['field'][shared], rows['energy'][shared], marker,
                              color=color, mfc='none', zorder=3, picker=5)
            self._pick_rows[artist] = rows[shared]
            bad = self._bad(rows)
            ax.plot(rows['field'][bad], rows['energy'][bad], 'x', color='black', ms=9, zorder=4)

            if fit is not None:
                b = np.linspace(rows['field'].min(), rows['field'].max(), 200)
                coeffs = [fit.get('b', 0.0), fit['a'], fit['E0']]
                ax.plot(b, np.polyval(coeffs, b), '--', color=color, lw=1, zorder=2)
            lines.append(self._fit_text(key, fit))

        ax.set_xlabel('Magnetic Field (T)')
        ax.set_ylabel('Peak energy (eV)')
        ax.set_title(f"Tracked peaks — session: {self.session}"
                     + ("" if self.include_shared else " (shared points excluded from fits)"),
                     fontsize=10)
        ax.legend(fontsize=7, loc='best')

        self.ax_text.cla()
        self.ax_text.axis('off')
        self.ax_text.text(0, 1, '\n'.join(lines) or 'No branches shown.', va='top',
                          family='monospace', fontsize=7.5, transform=self.ax_text.transAxes)
        self.fig.canvas.draw_idle()

    def _fit_text(self, key: tuple, fit: dict | None) -> str:
        label = self._label(key)
        if fit is None:
            return f"{label}:  too few points for a degree-{self.DEGREE} fit"
        parts = [f"E0 = {fit['E0']:.5f} ± {fit['E0_err']:.1e} eV",
                 f"a = {fit['a'] * 1e3:.4g} ± {fit['a_err'] * 1e3:.2g} meV/T"]
        if 'b' in fit:
            parts.append(f"b = {fit['b'] * 1e3:.4g} ± {fit['b_err'] * 1e3:.2g} meV/T²")
        return f"{label}:  " + ",  ".join(parts) + f"  (n = {fit['n_points']})"

    # ------------------------------------------------------------------
    # Internal: callbacks
    # ------------------------------------------------------------------

    def _on_session(self, session: str):
        self.session = session
        self._load_session()

    def _on_shared(self, _label):
        self.include_shared = not self.include_shared
        self._fit_all()
        self._draw()

    def _on_toggle(self, label):
        keys = self._keys()
        key = keys[[self._label(k) for k in keys].index(label)]
        self.visible[key] = not self.visible[key]
        self._draw()

    def _on_pick(self, event):
        rows = self._pick_rows.get(event.artist)
        if rows is None or event.mouseevent is self._last_click or len(event.ind) == 0:
            return
        self._last_click = event.mouseevent
        self._show_fit_plot(rows.iloc[event.ind[0]])

    def _fit_plot_path(self, row: pd.Series) -> Path:
        """Where PLFitter saved this spectrum's fit plot (see PLFitter._fits_dir)."""
        return (self.output_dir / str(row['experiment']) / 'plots'
                / f"fits_{int(row['fit_number'])}" / f"{Path(row['filename']).stem}_fit.png")

    def _show_fit_plot(self, row: pd.Series):
        path = self._fit_plot_path(row)
        if not path.exists():
            print(f"No saved fit plot: {path}")
            return
        if self._preview is None or not plt.fignum_exists(self._preview.number):
            self._preview = plt.figure(figsize=(8, 5))
        fig = self._preview
        fig.clf()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(mpimg.imread(path))
        ax.axis('off')
        fig.canvas.manager.set_window_title(  # pyright: ignore[reportOptionalMemberAccess]
            f"{row['experiment']} / {row['filename']} — fit {int(row['fit_number'])}")
        fig.show()
        fig.canvas.draw_idle()

    def _save(self, _event):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = f"_{self.session}"
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        bbox = self.ax.get_tightbbox(renderer).transformed(  # pyright: ignore[reportOptionalMemberAccess]
            self.fig.dpi_scale_trans.inverted())
        png = self.output_dir / f"branch_compare{stem}.png"
        self.fig.savefig(png, dpi=200, bbox_inches=bbox)

        rows = []
        for key, fit in self.fits.items():
            if not self.visible.get(key, True):
                continue
            rows.append({'experiment': key[0], 'branch': key[1],
                         'branch_e0': self._rows(key)['branch_e0'].iloc[0],
                         'session': self.session, 'include_shared': self.include_shared,
                         **(fit or {'n_points': None})})
        csv = self.output_dir / f"branch_fits{stem}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"Saved -> {png}\nSaved -> {csv}")
