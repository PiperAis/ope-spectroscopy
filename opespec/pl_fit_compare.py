"""
Cross-dataset comparison of saved PL fits (fit_results.csv from PLFitter):
follow fitted peaks through a field sweep and fit their energy trend.

Key components:
- track_branches: follow every peak of the seed fit (B = 0, or the lowest
  temperature) through each sweep (experiment), nearest energy step to
  step; flags points shared by more than one branch (merged peaks)
- fit_branch: weighted polynomial E(B) = E0 + a·B (+ b·B²) for one branch
- FitCompareViewer: interactive window — tracked branches (energy vs field)
  per experiment with error bars, shared/bad-fit flags and trend fits;
  branch toggles, fit-session picker, faded overlay of a second session,
  save PNG + coefficient CSV; click a point to view that spectrum's saved
  fit plot

Rules: works only on the fit_results DataFrame (load_fit_results) — no
refitting. A "series" is an experiment (sweep folder). The sweep variable
is passed explicitly (as in pl_sweep), never inferred. Each spectrum's
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
from .pl_sweep import _check_sweep


def track_branches(df: pd.DataFrame, fit_number: int | None = None,
                   sweep: str = 'field') -> pd.DataFrame:
    """
    Follow every peak of the seed spectrum's fit through each sweep.

    sweep : 'field' (default) or 'temperature', the column walked along.

    Per experiment, the seeds are all peaks of one spectrum: the one at
    B = 0 for a field sweep, or at the lowest temperature for a temperature
    sweep. Warns if a field sweep has no B = 0 spectrum (seeds at the field
    nearest 0 instead), or if more than one spectrum sits at the seed point
    (the first by filename seeds; the others are matched to its peaks like
    any other spectrum). Each seed is walked
    outward through the sweep values (both directions for field), taking at
    each step the peak nearest in energy to the one tracked at the previous
    step (as track_peak does for the brightest peak only).

    Returns one row per (branch, spectrum) with added columns:
      branch     0, 1, ... in order of seed energy, per experiment
      branch_e0  the seed energy (eV)
      shared     True where this fitted peak is also on another branch
    """
    units = _check_sweep(sweep)['units']
    data = _latest_saves(df, fit_number)
    tracked = []
    for experiment, points in data.groupby('experiment', dropna=False):
        points = points.dropna(subset=[sweep])
        if points.empty:
            continue
        values = np.sort(points[sweep].unique())
        if sweep == 'field':
            f0 = values[np.argmin(np.abs(values))]
            if f0 != 0:
                warnings.warn(f"{experiment}: no B = 0 spectrum; seeding branches at {f0:g} T.")
        else:
            f0 = values[0]
        at_f0 = points[points[sweep] == f0]
        seed_files = sorted(at_f0['filename'].unique())
        if len(seed_files) > 1:
            warnings.warn(f"{experiment}: {len(seed_files)} spectra at the seed point "
                          f"({f0:g} {units}): {', '.join(seed_files)}. Seeding from {seed_files[0]}.")
        seeds = np.sort(at_f0[at_f0['filename'] == seed_files[0]]['energy'].to_numpy())

        for branch, e0 in enumerate(seeds):
            for direction in (values[values > f0], values[values < f0][::-1], [f0]):
                ref = e0
                for f in direction:
                    chosen = []
                    for _, spectrum in points[points[sweep] == f].groupby('filename'):
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
    return out.sort_values(['experiment', 'branch', sweep]).reset_index(drop=True)


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

    - Grey points: the fitted peaks of the selected session (context). Peaks
      only on hidden branches are left out; an experiment's peaks on no
      branch show only while all of its branches are shown.
    - Branches: colour = experiment, marker = branch, error bars = energy_err.
      Hollow markers are shared with another branch (merged peaks); a black ×
      flags a doubtful fit (r_squared < R2_MIN, or energy_err/width > REL_ERR_MAX).
    - Dashed line: fit_branch trend (quadratic); coefficients listed below.

    Controls: one checkbox per experiment (ticked while any of its branches
    is shown; a click hides all of them, or shows all if none were shown);
    one checkbox per (experiment, branch); "Session" cycles
    latest → each fit_number; "Overlay" cycles none → latest → each
    fit_number; "Include shared points in trend fit"; Save.
    The overlay session is tracked and fitted on its own and drawn faded
    (same colours and markers, its fit lines listed after the main ones as
    "[overlay <session>]"). Branches are matched between sessions by
    (experiment, branch index), so one checkbox toggles both; check the
    E at B≈0 in the two coefficient lines to confirm they are the same peak.
    Clicking any point (branch or grey) shows that spectrum's saved fit plot,
    <output_dir>/<experiment>/plots/fits_<fit_number>/<stem>_fit.png, in a
    separate window that is reused by later clicks.

    Save writes <output_dir>/branch_compare_<session>.png and
    <output_dir>/branch_fits_<session>.csv (coefficients of the visible
    branches; energies in eV, a in eV/T, b in eV/T²). With an overlay the
    names end _<session>_vs_<overlay> and the CSV also holds the overlay's
    rows (its 'session' column tells them apart). Files are overwritten by a
    later save of the same session (and overlay).

    Must be run as a plain script (like PLFitter), not in a #%% cell.

    Usage:
        df = load_fit_results(Path(cfg['processed_pl_dir']) / 'fit_results.csv')
        FitCompareViewer(df, output_dir=Path(cfg['processed_pl_dir'])).run()
    """

    R2_MIN = 0.95
    REL_ERR_MAX = 0.5   # energy_err / width: centre not resolved within the line
    DEGREE = 2
    OVERLAY_ALPHA = 0.3
    _MARKERS = 'osD^v<>ph*'

    def __init__(self, df: pd.DataFrame, output_dir: Path):
        self.df = df
        self.output_dir = Path(output_dir)
        sessions = sorted(int(n) for n in df['fit_number'].dropna().unique())
        self._sessions = ['latest'] + [str(n) for n in sessions]
        self.session = 'latest'
        self.overlay = 'none'
        self.include_shared = False
        self.visible: dict[tuple, bool] = {}   # (experiment, branch) → shown; persists across sessions
        self.experiments = sorted(df['experiment'].dropna().astype(str).unique())

        self.fig: Figure  # created in run()
        self.ax: Axes
        self.ax_experiments: Axes
        self.ax_toggles: Axes
        self.ax_text: Axes
        self.cb_experiments: CheckButtons | None = None
        self.cb_branches: CheckButtons | None = None
        self.branches = pd.DataFrame()
        self.fits: dict[tuple, dict | None] = {}
        self.ov_branches = pd.DataFrame()      # overlay session, same layout
        self.ov_fits: dict[tuple, dict | None] = {}
        self._pick_rows: dict = {}             # plotted point artist → its rows, for click-to-open
        self._last_click = None                # one click can pick several overlapping artists
        self._preview: Figure | None = None

    @property
    def fit_number(self) -> int | None:
        return None if self.session == 'latest' else int(self.session)

    @property
    def overlay_fit_number(self) -> int | None:
        return None if self.overlay in ('latest', 'none') else int(self.overlay)

    def run(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes((0.08, 0.42, 0.60, 0.53))
        self.ax_experiments = self.fig.add_axes((0.72, 0.80, 0.27, 0.15))
        self.ax_toggles = self.fig.add_axes((0.72, 0.42, 0.27, 0.34))
        self.ax_text = self.fig.add_axes((0.02, 0.02, 0.96, 0.20))
        self.ax_text.axis('off')

        ax_session = self.fig.add_axes((0.08, 0.25, 0.13, 0.05))
        ax_overlay = self.fig.add_axes((0.22, 0.25, 0.13, 0.05))
        ax_shared = self.fig.add_axes((0.36, 0.24, 0.21, 0.07))
        ax_save = self.fig.add_axes((0.58, 0.25, 0.10, 0.05))
        self.btn_session = _CycleButton(ax_session, 'Session', self._sessions, self.session,
                                        on_change=self._on_session)
        self.btn_overlay = _CycleButton(ax_overlay, 'Overlay', ['none', *self._sessions],
                                        self.overlay, on_change=self._on_overlay)
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
        """Re-track branches for the selected (and overlay) session and rebuild the toggles."""
        self.branches = self._track(self.fit_number)
        self.ov_branches = (pd.DataFrame() if self.overlay == 'none'
                            else self._track(self.overlay_fit_number))
        self._fit_all()
        self._build_toggles()
        self._draw()

    def _track(self, fit_number: int | None) -> pd.DataFrame:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            branches = track_branches(self.df, fit_number)
        for w in caught:
            print(f"Warning: {w.message}")
        return branches

    def _fit_all(self):
        self.fits = self._fit_branches(self.branches)
        self.ov_fits = self._fit_branches(self.ov_branches)

    def _fit_branches(self, branches: pd.DataFrame) -> dict[tuple, dict | None]:
        if branches.empty:
            return {}
        return {key: fit_branch(rows, self.DEGREE, self.include_shared)
                for key, rows in branches.groupby(['experiment', 'branch'])}

    def _keys(self) -> list[tuple]:
        """Toggle keys: every branch in the main or overlay session, by experiment then branch."""
        return sorted({*self.fits, *self.ov_fits}, key=lambda k: (str(k[0]), k[1]))

    def _rows(self, key: tuple, branches: pd.DataFrame | None = None) -> pd.DataFrame:
        branches = self.branches if branches is None else branches
        if branches.empty:
            return branches
        exp, br = key
        return branches[(branches['experiment'] == exp)
                        & (branches['branch'] == br)].sort_values('field')

    def _label(self, key: tuple, branches: pd.DataFrame | None = None) -> str:
        """'<experiment> · <seed energy>'. With no branches given (a toggle), taken
        from the main session if it has the key, else the overlay."""
        rows = self._rows(key, branches)
        if rows.empty and branches is None:
            rows = self._rows(key, self.ov_branches)
        return f"{key[0]} · {rows['branch_e0'].iloc[0]:.4f} eV"

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

    def _experiment_keys(self) -> dict:
        """experiment → its toggle keys, in toggle order."""
        groups: dict = {}
        for k in self._keys():
            groups.setdefault(k[0], []).append(k)
        return groups

    def _build_toggles(self):
        # Old widgets stay hooked to the canvas (clicks, redraws) after cla(); unhook them.
        for cb in (self.cb_experiments, self.cb_branches):
            if cb is not None:
                cb.disconnect_events()
        self.ax_experiments.cla()
        self.ax_toggles.cla()
        keys = self._keys()
        for k in keys:
            self.visible.setdefault(k, True)
        if not keys:
            self.ax_experiments.axis('off')
            self.ax_toggles.axis('off')
            self.cb_experiments = None
            self.cb_branches = None
            return
        self.ax_experiments.axis('on')
        self.ax_toggles.axis('on')

        groups = self._experiment_keys()
        self.cb_experiments = CheckButtons(
            self.ax_experiments, [str(e) for e in groups],
            [any(self.visible[k] for k in ks) for ks in groups.values()])
        for text in self.cb_experiments.labels:
            text.set_fontsize(8)
        self.cb_experiments.on_clicked(self._on_experiment)
        self.ax_experiments.set_title('Experiments (all branches)', fontsize=9)

        self.cb_branches = CheckButtons(self.ax_toggles, [self._label(k) for k in keys],
                                        [self.visible[k] for k in keys])
        for text in self.cb_branches.labels:
            text.set_fontsize(8)
        self.cb_branches.on_clicked(self._on_toggle)
        self.ax_toggles.set_title('Branches (experiment · E at B≈0)', fontsize=9)

    def _context(self) -> pd.DataFrame:
        """Grey context points: the session's fitted peaks that are on a shown
        branch, plus an experiment's peaks on no branch only while all of that
        experiment's branches (in this session) are shown."""
        context = _latest_saves(self.df, self.fit_number).dropna(subset=['field'])
        if self.branches.empty:
            return context
        peak_id = ['experiment', 'filename', 'peak_index']
        b = self.branches
        shown = np.array([self.visible.get((e, br), True)
                          for e, br in zip(b['experiment'], b['branch'])], dtype=bool)
        on_shown = set(map(tuple, b.loc[shown, peak_id].to_numpy()))
        on_any = set(map(tuple, b[peak_id].to_numpy()))
        all_shown = {e for e, s in pd.Series(shown, index=b.index).groupby(b['experiment']).all()
                     .items() if s}
        keep = [p in on_shown or (p not in on_any and p[0] in all_shown)
                for p in map(tuple, context[peak_id].to_numpy())]
        return context[np.array(keep, dtype=bool)]

    def _draw(self):
        ax = self.ax
        ax.cla()
        self._pick_rows = {}
        context = self._context()
        artist, = ax.plot(context['field'], context['energy'], '.', color='0.8', zorder=1,
                          label='Fitted peaks', picker=5)
        self._pick_rows[artist] = context

        lines = self._draw_branches(self.branches, self.fits)
        if self.overlay != 'none':
            ax.plot([], [], 'o', color='0.4', alpha=self.OVERLAY_ALPHA,
                    label=f"Overlay: session {self.overlay}")
            lines += self._draw_branches(self.ov_branches, self.ov_fits, overlay=True)

        ax.set_xlabel('Magnetic Field (T)')
        ax.set_ylabel('Peak energy (eV)')
        ax.set_title(f"Tracked peaks — session: {self.session}"
                     + ("" if self.overlay == 'none' else f", overlay: {self.overlay} (faded)")
                     + ("" if self.include_shared else " (shared points excluded from fits)"),
                     fontsize=10)
        ax.legend(fontsize=7, loc='best')

        self.ax_text.cla()
        self.ax_text.axis('off')
        self.ax_text.text(0, 1, '\n'.join(lines) or 'No branches shown.', va='top',
                          family='monospace', fontsize=7.5, transform=self.ax_text.transAxes)
        self.fig.canvas.draw_idle()

    def _draw_branches(self, branches: pd.DataFrame, fits: dict,
                       overlay: bool = False) -> list[str]:
        """Draw one session's visible branches; return their coefficient lines.
        The overlay is faded, drawn under the main session and kept out of the legend."""
        ax = self.ax
        alpha = self.OVERLAY_ALPHA if overlay else 1.0
        z = -0.5 if overlay else 0
        lines = []
        for key, fit in fits.items():
            if not self.visible.get(key, True):
                continue
            rows = self._rows(key, branches)
            color = self._color(key[0])
            marker = self._MARKERS[int(key[1]) % len(self._MARKERS)]
            shared = rows['shared'].astype(bool)
            ax.errorbar(rows['field'], rows['energy'], yerr=rows['energy_err'],
                        color=color, lw=0.8, capsize=2, marker='none', alpha=alpha, zorder=2 + z)
            artist, = ax.plot(rows['field'][~shared], rows['energy'][~shared], marker,
                              color=color, alpha=alpha, zorder=3 + z, picker=5,
                              label=None if overlay else self._label(key, branches))
            self._pick_rows[artist] = rows[~shared]
            artist, = ax.plot(rows['field'][shared], rows['energy'][shared], marker,
                              color=color, mfc='none', alpha=alpha, zorder=3 + z, picker=5)
            self._pick_rows[artist] = rows[shared]
            bad = self._bad(rows)
            ax.plot(rows['field'][bad], rows['energy'][bad], 'x', color='black', ms=9,
                    alpha=alpha, zorder=4 + z)

            if fit is not None:
                b = np.linspace(rows['field'].min(), rows['field'].max(), 200)
                coeffs = [fit.get('b', 0.0), fit['a'], fit['E0']]
                ax.plot(b, np.polyval(coeffs, b), '--', color=color, lw=1, alpha=alpha,
                        zorder=2 + z)
            prefix = f"[overlay {self.overlay}] " if overlay else ""
            lines.append(prefix + self._fit_text(key, fit, branches))
        return lines

    def _fit_text(self, key: tuple, fit: dict | None,
                  branches: pd.DataFrame | None = None) -> str:
        label = self._label(key, branches)
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

    def _on_overlay(self, overlay: str):
        self.overlay = overlay
        self._load_session()

    def _on_shared(self, _label):
        self.include_shared = not self.include_shared
        self._fit_all()
        self._draw()

    def _on_toggle(self, label):
        keys = self._keys()
        key = keys[[self._label(k) for k in keys].index(label)]
        self.visible[key] = not self.visible[key]
        self._build_toggles()   # the experiment's box may change with it
        self._draw()

    def _on_experiment(self, label):
        keys = next(ks for e, ks in self._experiment_keys().items() if str(e) == label)
        show = not any(self.visible[k] for k in keys)
        for k in keys:
            self.visible[k] = show
        self._build_toggles()
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
        stem = f"_{self.session}" + ("" if self.overlay == 'none' else f"_vs_{self.overlay}")
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue]
        bbox = self.ax.get_tightbbox(renderer).transformed(  # pyright: ignore[reportOptionalMemberAccess]
            self.fig.dpi_scale_trans.inverted())
        png = self.output_dir / f"branch_compare{stem}.png"
        self.fig.savefig(png, dpi=200, bbox_inches=bbox)

        rows = []
        runs = [(self.session, self.branches, self.fits)]
        if self.overlay != 'none':
            runs.append((self.overlay, self.ov_branches, self.ov_fits))
        for session, branches, fits in runs:
            for key, fit in fits.items():
                if not self.visible.get(key, True):
                    continue
                rows.append({'experiment': key[0], 'branch': key[1],
                             'branch_e0': self._rows(key, branches)['branch_e0'].iloc[0],
                             'session': session, 'include_shared': self.include_shared,
                             **(fit or {'n_points': None})})
        csv = self.output_dir / f"branch_fits{stem}.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"Saved -> {png}\nSaved -> {csv}")
