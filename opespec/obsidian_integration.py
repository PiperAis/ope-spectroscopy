"""
Obsidian vault integration.

TRR:
- generate_dataset_notes: write one .md note per processed .nc file,
  populated with frontmatter from the filename and the experiment note
  (the report folder itself is copied by trr_driver.py)

Steady-state PL:
- sync_pl_sweep: one 'Dataset N.md' note per fitted spectrum in a processed
  sweep folder, in <vault project>/<sweep> PL/, with metadata resolved like
  load_pl_data (filename + raw folder metadata.txt, via get_metadata) and the
  spectrum's latest fit plot copied in and embedded. Existing
  notes keep their body and hand-added properties; only generated
  properties are rewritten.

Vault locations always come in as arguments (from config), never built here
beyond the folder-naming conventions documented on each function.
"""
from pathlib import Path
import re
import shutil
import xarray as xr
import yaml

from .utilities import get_field, get_metadata, get_sample_name

def _parse_frontmatter(md_path: Path) -> dict:
    """Extract YAML frontmatter from a markdown file (between --- delimiters)."""
    text = md_path.read_text(encoding='utf-8')
    if not text.startswith('---'):
        return {}
    parts = text.split('---', 2)
    if len(parts) < 3:
        return {}
    return yaml.safe_load(parts[1]) or {}


# Regex patterns for known token types (matched against individual '-'-split tokens)
_KNOWN_TOKEN_PATTERNS = [
    re.compile(r'^flake'),            # sample name
    re.compile(r'^n?\d+(p\d+)?T$'),  # B-field
    re.compile(r'^\d+p?\d*K$'),      # temperature
    re.compile(r'^\d+kSF$'),         # scale factor
    re.compile(r'^\d+p?(\d+)?mV$'),  # r0
]

# Experiment-note properties that are never relevant at the dataset level
_EXPT_ONLY_KEYS = {'Datasets', 'Summary', 'Detection notes'}


def generate_dataset_notes(processed_data_dir: Path, vault_expt_dir: Path, expt_name: str) -> None:
    """Write one Obsidian .md note per processed .nc dataset into vault_expt_dir."""

    # Source 1: experiment-note frontmatter
    expt_note = vault_expt_dir / f'{expt_name}.md'
    expt_props = _parse_frontmatter(expt_note) if expt_note.exists() else {}

    for nc_file in sorted(processed_data_dir.glob('*.nc')):
        stem = nc_file.stem  # e.g. "01-flake1a-1p6K-0p2T-70kSF-120mV-2pass-offset1"
        tokens = stem.split('-')
        set_number = tokens[0]

        # Source 2: known filename params via existing parsers
        temp_match = re.search(r'-(\d+p?\d*)K', stem)
        temperature = float(temp_match.group(1).replace('p', '.')) if temp_match else None

        filename_props = {
            'Sample':      get_sample_name(stem),
            'Bfield':      get_field(stem),
        }
        if temperature is not None:
            filename_props['Temperature'] = temperature

        # Source 3: leftover tokens (skip index 0 = set number)
        other_params = [
            tok for tok in tokens[1:]
            if not any(p.match(tok) for p in _KNOWN_TOKEN_PATTERNS)
        ]

        # Filter experiment props: include only if the key also appears in the
        # filename (filename value will win) OR the value is 'Variable' or a
        # list (meaning it differs per dataset). Drop experiment-only keys.
        filtered_expt = {
            k: v for k, v in expt_props.items()
            if k not in _EXPT_ONLY_KEYS
            and (k in filename_props or v == 'Variable' or isinstance(v, list))
        }

        # Merge: filename_props wins over filtered experiment props
        props = {**filtered_expt, **filename_props}

        # Build frontmatter dict in desired order
        frontmatter_dict = {'Type': 'Dataset', **props,
                            'other_params': other_params,
                            'quickplot': f'[[{set_number}-processed.png]]'}

        fm_str = yaml.dump(frontmatter_dict, default_flow_style=False,
                           allow_unicode=True, sort_keys=False)
        note_content = f'---\n{fm_str}Notes:\n---\n'

        (vault_expt_dir / f'{set_number}.md').write_text(note_content, encoding='utf-8')


# ---------------------------------------------------------------------------
# Steady-state PL
# ---------------------------------------------------------------------------

# Metadata key → note property, in note order. These are always rewritten on sync.
_PL_ATTR_PROPS = (('sample', 'Sample'), ('spot', 'Spot'),
                  ('field', 'Bfield'), ('temperature', 'Temperature'))
# get_metadata keys that are TRR concepts; never written to PL notes.
_PL_SKIP_KEYS = {'scale_factor', 'rzero'}


def _split_note(md_path: Path) -> tuple[dict, str]:
    """(frontmatter dict, body text) of an existing note; ({}, text) if it has none."""
    text = md_path.read_text(encoding='utf-8')
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) == 3:
            return (yaml.safe_load(parts[1]) or {}), parts[2].lstrip('\n')
    return {}, text


def _write_note(md_path: Path, props: dict, body: str) -> None:
    fm = yaml.dump(props, default_flow_style=False, allow_unicode=True, sort_keys=False)
    md_path.write_text(f'---\n{fm}---\n{body}', encoding='utf-8')


def _attr_value(value):
    """netCDF attr → plain YAML-safe value (numpy scalars → Python)."""
    return value.item() if hasattr(value, 'item') else value


def _pl_metadata(raw_sweep_dir: Path, attrs: dict, stem: str) -> dict:
    """
    Note properties from the spectrum's metadata, resolved like load_pl_data:
    utilities.get_metadata on the raw file (filename wins over the raw folder's
    metadata.txt), so extra metadata.txt keys come through too. If the raw
    folder isn't available locally, falls back to the .nc attrs (standard keys only).
    """
    if raw_sweep_dir.is_dir():
        metadata = get_metadata(raw_sweep_dir / attrs.get('filename', f'{stem}.csv'))
    else:
        metadata = {k: _attr_value(attrs[k]) for k, _ in _PL_ATTR_PROPS if k in attrs}
    standard = dict(_PL_ATTR_PROPS)
    props = {prop: metadata[key] for key, prop in _PL_ATTR_PROPS
             if metadata.get(key) is not None}
    props.update({k: v for k, v in metadata.items()
                  if k not in standard and k not in _PL_SKIP_KEYS})
    return props


def sync_pl_sweep(processed_sweep_dir: Path, raw_sweep_dir: Path,
                  vault_project_dir: Path) -> Path:
    """
    Write/update one Obsidian note per fitted spectrum (.nc) in a processed
    PL sweep folder. Returns the vault experiment folder.

    Layout (matches the hand-made vault notes):
        <vault_project_dir>/<sweep> PL/<sweep> PL.md      experiment note (read only)
        <vault_project_dir>/<sweep> PL/Dataset <N>.md     N = the file's ## prefix
        <vault_project_dir>/<sweep> PL/plots/<sweep>_<stem>_fit.png

    Metadata comes from the raw spectrum in raw_sweep_dir via get_metadata
    (filename, then metadata.txt): Sample, Spot, Bfield, Temperature, plus any
    other metadata.txt keys under their own names. See _pl_metadata.

    The plot is the spectrum's latest saved fit, plots/fits_<fit_number>/<stem>_fit.png
    (fit_number from the .nc attrs); its vault name includes the sweep so wikilinks
    stay unique across experiments.

    Existing notes: generated properties (Type, Experiment, metadata, Filename,
    quickplot) are rewritten; every other property and the note body are kept.
    Properties inherited from the experiment note ('Variable' or list values)
    and an empty Notes property are only added when missing.
    """
    processed_sweep_dir = Path(processed_sweep_dir)
    raw_sweep_dir = Path(raw_sweep_dir)
    vault_project_dir = Path(vault_project_dir)
    if not vault_project_dir.is_dir():
        raise FileNotFoundError(f"Vault project folder not found: {vault_project_dir}")
    if not raw_sweep_dir.is_dir():
        print(f"  Raw folder not found ({raw_sweep_dir}); using .nc metadata only.")

    sweep = processed_sweep_dir.name
    expt_name = f'{sweep} PL'
    expt_dir = vault_project_dir / expt_name
    plots_dir = expt_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    expt_note = expt_dir / f'{expt_name}.md'
    expt_props = _parse_frontmatter(expt_note) if expt_note.exists() else {}
    inherited = {k: v for k, v in expt_props.items()
                 if k not in _EXPT_ONLY_KEYS and (v == 'Variable' or isinstance(v, list))}

    nc_files = sorted(processed_sweep_dir.glob('*.nc'))
    prefixes = [f.stem.split('-')[0] for f in nc_files]
    for nc_file, prefix in zip(nc_files, prefixes):
        stem = nc_file.stem
        with xr.open_dataset(nc_file) as ds:
            attrs = dict(ds.attrs)

        # Note name from the ## prefix; the full stem if the prefix is missing or shared.
        if prefix.isdigit() and prefixes.count(prefix) == 1:
            note_name = f'Dataset {int(prefix)}'
        else:
            note_name = f'Dataset {stem}'

        # Latest fit plot → vault
        quickplot = None
        fit_number = attrs.get('fit_number')
        if fit_number is not None:
            src = processed_sweep_dir / 'plots' / f'fits_{int(fit_number)}' / f'{stem}_fit.png'
            if src.exists():
                dest_name = f'{sweep}_{stem}_fit.png'
                shutil.copy2(src, plots_dir / dest_name)
                quickplot = dest_name
            else:
                print(f"  No fit plot for {stem}: {src}")
        else:
            print(f"  {stem}.nc has no fit_number; no plot linked.")

        generated = {'Type': 'Dataset', 'Experiment': f'[[{expt_name}]]',
                     **_pl_metadata(raw_sweep_dir, attrs, stem),
                     'Filename': stem}
        if quickplot:
            generated['quickplot'] = f'[[{quickplot}]]'

        note_path = expt_dir / f'{note_name}.md'
        if note_path.exists():
            props, body = _split_note(note_path)
            props.update(generated)
        else:
            props = dict(generated)
            body = f'# Quickplot\n![[{quickplot}]]\n' if quickplot else ''
        for k, v in inherited.items():
            props.setdefault(k, v)
        props.setdefault('Notes', None)

        _write_note(note_path, props, body)

    return expt_dir