"""
One-off migration: processed PL .nc files from the one-fit layout (fit
settings in attrs, overwritten by every save) to the session layout (every
fit session of a spectrum along a fit_number dimension; see the processed-
spectrum section of opespec.steady_state).

Per <processed_pl_dir>/<experiment>/<stem>.nc:
  - the fit stored in the .nc becomes session <its fit_number>, exactly (0 if
    it predates session numbering), with n_points / dof from fit_results.csv;
  - every other session of that spectrum in fit_results.csv is added from the
    CSV rows: peaks and settings only, with no baseline (NaN), so those
    sessions need a refit to be edited.
fit_results.csv is then regenerated from the .nc files.

Dry run by default: converts in memory and checks that
  - each .nc's own fit (raw, baseline, subtracted, settings, peaks) is unchanged;
  - every fit_results.csv row is reproduced by the regenerated CSV;
and reports anything that would not carry over. --write then backs up the
.nc files to <processed_pl_dir>/pre_migration_backup/<experiment>/ and the CSV
to fit_results_pre_migration.csv, and writes the converted files.

    python scripts/migrate_pl_sessions.py --project HS-2            # dry run
    python scripts/migrate_pl_sessions.py --project HS-2 --write
    python scripts/migrate_pl_sessions.py --dir <processed_pl_dir> [--write]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from opespec.utilities import load_config
from opespec.steady_state import (_session_dataset, _put_session, _session_attrs, _session_rows,
                                  _write_fit_results, _PEAK_KEYS, _METADATA_KEYS)

BACKUP_DIR = 'pre_migration_backup'
KEYS = ['experiment', 'filename', 'fit_number', 'peak_index']


def truthy(value) -> bool:
    return str(value).strip().lower() in ('true', '1', '1.0')


def nan_if_none(value):
    return np.nan if value is None or (isinstance(value, float) and np.isnan(value)) else value


def session_from_attrs(attrs: dict, csv_rows: pd.DataFrame) -> tuple[dict, dict]:
    """(settings, peaks) of the fit stored in an old one-fit .nc's attrs."""
    settings = {k: attrs.get(k) for k in ('lineshape', 'baseline_type', 'baseline_c0',
                                          'baseline_c1', 'baseline_xc', 'fit_emin', 'fit_emax',
                                          'rmse', 'r_squared', 'spacing', 'spacing_err')}
    settings['shared_width_locked'] = int(attrs.get('shared_width_locked', 0))
    if not csv_rows.empty:   # not stored in the .nc
        settings['n_points'] = csv_rows['n_points'].iloc[0]
        settings['dof'] = csv_rows['dof'].iloc[0]
    k = len(np.atleast_1d(attrs['peak_energy']))
    peaks = {key: np.atleast_1d(attrs.get(f'peak_{key}', np.full(k, np.nan))).astype(float)
             for key in _PEAK_KEYS}
    peaks['width_shared'] = np.atleast_1d(attrs.get('peak_width_shared', np.zeros(k))).astype(bool)
    peaks['energy_spaced'] = np.atleast_1d(attrs.get('peak_energy_spaced', np.zeros(k))).astype(bool)
    return settings, peaks


def session_from_csv(rows: pd.DataFrame) -> tuple[dict, dict]:
    """(settings, peaks) of a session known only from its fit_results.csv rows."""
    rows = rows.sort_values('peak_index')
    first = rows.iloc[0]

    def col(name):
        return rows[name] if name in rows else pd.Series(np.nan, index=rows.index)

    spacing = col('spacing').dropna()
    spacing_err = col('spacing_err').dropna()
    settings = {
        'lineshape':     first.get('lineshape'),
        'baseline_type': first.get('baseline'),
        'fit_emin':      nan_if_none(first.get('fit_emin')),
        'fit_emax':      nan_if_none(first.get('fit_emax')),
        'rmse':          nan_if_none(first.get('rmse')),
        'r_squared':     nan_if_none(first.get('r_squared')),
        'n_points':      nan_if_none(first.get('n_points')),
        'dof':           nan_if_none(first.get('dof')),
        'spacing':       spacing.iloc[0] if len(spacing) else None,
        'spacing_err':   spacing_err.iloc[0] if len(spacing_err) else None,
        'shared_width_locked': int(any(truthy(v) for v in col('width_locked'))),
    }
    peaks = {key: col(key).astype(float).values for key in _PEAK_KEYS}
    peaks['width_shared'] = [truthy(v) for v in col('width_shared')]
    peaks['energy_spaced'] = [truthy(v) for v in col('energy_spaced')]
    return settings, peaks


def convert(nc: Path, old: xr.Dataset, csv: pd.DataFrame) -> tuple[xr.Dataset, int]:
    """The session-layout Dataset for one old .nc, and the fit_number of its own fit."""
    attrs = dict(old.attrs)
    experiment = str(attrs.get('experiment', nc.parent.name))
    n = int(attrs.get('fit_number', 0))
    raw = old['raw']
    mine = csv[(csv['experiment'] == experiment) & (csv['filename'] == attrs['filename'])]

    settings, peaks = session_from_attrs(attrs, mine[mine['fit_number'] == n])
    ds = _put_session(None, raw, _session_dataset(n, raw, settings, peaks, old['baseline'].values))
    for m, rows in mine[mine['fit_number'] != n].groupby('fit_number'):
        settings, peaks = session_from_csv(rows)
        ds = _put_session(ds, raw, _session_dataset(int(m), raw, settings, peaks, None))
    ds.attrs = {**{k: attrs[k] for k in ('filename', *_METADATA_KEYS) if k in attrs},
                'experiment': experiment}
    return ds, n


def check_own_fit(name: str, old: xr.Dataset, new: xr.Dataset, n: int) -> list[str]:
    """Differences between an old .nc's fit and session n of its conversion."""
    problems = []
    for var in ('raw', 'baseline', 'subtracted'):
        a = old[var].values
        b = new[var].values if var == 'raw' else new[var].sel(fit_number=n).values
        if not np.allclose(a, b, equal_nan=True):
            problems.append(f"{name}: {var} differs")
    attrs = _session_attrs(new, n)
    for key, value in old.attrs.items():
        if key in ('filename', 'experiment', *_METADATA_KEYS):
            if new.attrs.get(key) != value:
                problems.append(f"{name}: attr {key} {value!r} -> {new.attrs.get(key)!r}")
            continue
        if key not in attrs:
            problems.append(f"{name}: attr {key} not carried over")
        elif isinstance(value, str) or isinstance(attrs[key], str):
            if str(value) != str(attrs[key]):
                problems.append(f"{name}: attr {key} {value!r} -> {attrs[key]!r}")
        elif not np.allclose(np.atleast_1d(value).astype(float),
                             np.atleast_1d(attrs[key]).astype(float), equal_nan=True):
            problems.append(f"{name}: attr {key} differs")
    return problems


def check_csv(old_csv: pd.DataFrame, new_csv: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Old CSV rows that the regenerated CSV does not reproduce (problems), and
    old rows with no .nc to carry them (missing)."""
    merged = old_csv.merge(new_csv, on=KEYS, how='left', suffixes=('', '_new'), indicator=True)
    missing = merged[merged['_merge'] == 'left_only'][KEYS]
    found = merged[merged['_merge'] == 'both']
    problems = []
    for col in old_csv.columns:
        if col in KEYS or f'{col}_new' not in found:
            continue
        a, b = found[col], found[f'{col}_new']
        if col in ('width_shared', 'width_locked', 'energy_spaced'):
            bad = a.map(truthy) != b.map(truthy)
        else:
            an, bn = pd.to_numeric(a, errors='coerce'), pd.to_numeric(b, errors='coerce')
            numeric = an.notna() | bn.notna()
            close = np.isclose(an.astype(float), bn.astype(float), rtol=1e-9, atol=0, equal_nan=True)
            bad = (numeric & ~close) | (~numeric & (a.isna() != b.isna()))
            bad |= ~numeric & a.notna() & (a.astype(str) != b.astype(str))
        for _, row in found[bad].head(3).iterrows():
            problems.append(f"CSV {row['experiment']}/{row['filename']} fit {row['fit_number']} "
                            f"peak {row['peak_index']}: {col} {row[col]!r} -> {row[f'{col}_new']!r}")
        if bad.sum() > 3:
            problems.append(f"  ... {bad.sum() - 3} more {col} differences")
    return problems, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument('--project', help='project name (uses its processed_pl_dir)')
    where.add_argument('--dir', type=Path, help='a processed_pl_dir to migrate directly')
    parser.add_argument('--write', action='store_true', help='write the converted files')
    args = parser.parse_args()

    out = args.dir if args.dir else Path(load_config(args.project)['processed_pl_dir'])
    csv_path = out / 'fit_results.csv'
    csv = pd.DataFrame(columns=KEYS)
    if csv_path.exists():
        csv = pd.read_csv(csv_path, dtype={'experiment': str, 'filename': str})
        if 'saved_at' in csv.columns:
            sys.exit(f"{csv_path} already has the new layout (saved_at column); nothing to do.")
        if 'experiment' not in csv.columns or 'fit_number' not in csv.columns:
            print(f"{csv_path.name} has no experiment/fit_number columns; "
                  f"only the fits stored in the .nc files can be migrated.")
            csv = pd.DataFrame(columns=KEYS)

    converted, problems = {}, []
    for nc in sorted(out.glob('*/*.nc')):
        name = f"{nc.parent.name}/{nc.name}"
        old = xr.load_dataset(nc)
        if 'fit_number' in old.dims:
            print(f"  {name}: already converted")
            continue
        if 'baseline' not in old.data_vars:
            print(f"  {name}: no fit stored; left as is")
            continue
        new, n = convert(nc, old, csv)
        converted[nc] = new
        problems += check_own_fit(name, old, new, n)
        sessions = [int(v) for v in new['fit_number'].values]
        print(f"  {name}: sessions {sessions} (own fit: {n})")
    if not converted:
        sys.exit("No .nc files to convert.")

    new_rows = []
    for nc, ds in converted.items():
        new_rows += _session_rows(ds, ds.attrs['experiment'])
    new_csv = pd.DataFrame(new_rows)
    if len(csv):
        csv_problems, missing = check_csv(csv, new_csv)
        problems += csv_problems
        added = new_csv.merge(csv[KEYS], on=KEYS, how='left', indicator=True)
        added = added[added['_merge'] == 'left_only']
    else:
        missing, added = pd.DataFrame(columns=KEYS), new_csv

    print(f"\n{len(converted)} .nc files; {len(csv)} old CSV rows -> {len(new_csv)} regenerated rows.")
    if len(added):
        print("Rows new to fit_results.csv (fits that were only in a .nc):")
        print(added.groupby(['experiment', 'fit_number'])['filename'].unique().to_string())
    if len(missing):
        print(f"Old CSV rows with no .nc to hold them ({len(missing)}); they will not be in the "
              f"regenerated CSV (kept in fit_results_pre_migration.csv):")
        print(missing.groupby(['experiment', 'fit_number'])['filename'].unique().to_string())
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        print('\n'.join(problems))
        sys.exit("Not writing: fix the problems above first.")
    print("\nChecks passed: every .nc's own fit is unchanged and every CSV row with a .nc is reproduced.")

    if not args.write:
        print("Dry run; rerun with --write to convert.")
        return
    for nc, ds in converted.items():
        backup = out / BACKUP_DIR / nc.parent.name / nc.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(nc, backup)
        ds.to_netcdf(nc, mode='w')
    print(f"Wrote {len(converted)} .nc files (originals in {out / BACKUP_DIR}).")
    print(f"Regenerated {_write_fit_results(out)}")


if __name__ == '__main__':
    main()
