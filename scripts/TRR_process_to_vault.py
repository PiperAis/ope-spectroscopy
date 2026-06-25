'''
Runs TRR, creates a report, then copies the folder (report and images) to Obsidian vault.

Generates an Obsidian note for each dataset and fills in parameters.
'''
#%%
# ---- User Input ---- #

# Input the name of the folder containing the data to be processed.
# Assumes data folder is inside of ..\TRR\raw_data

expt_date = "test"

# ---- Import modules ---- #
import argparse
from pathlib import Path
import re
import shutil
import yaml

ROOT_DIR = Path(__file__).parent.parent
from processing_package import ProcessingState
from processing_package.utilities import get_field, get_sample_name, load_config


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run TRR processing pipeline.")
    parser.add_argument(
        '--project',
        default=None,
        help="Name of a directory in the same parent folder as this script containing config.yaml"
    )
    args, _ = parser.parse_known_args()

    if args.project:
        config_path = ROOT_DIR.parent / args.project / 'config.yaml'
    else:
        config_path = ROOT_DIR.parent / '07_CrSBr_Cryo4' / 'config.yaml'

    cfg = load_config(config_path)

    experiment = ProcessingState(experiment_name = expt_date,
                                   steps_list=['noise corrected',
                                               'normalized',
                                               'processed',
                                               'FFT'],
                                   project_config = cfg)

    # Generate outline for markdown report
    report_file = experiment.generate_report_skeleton()

    # --- Step 1: Remove noise --- #
    experiment.run_noise_remover()

    # --- Step 2: Processing (normalize and subtract decay) --- #

    experiment.normalize_data()
    experiment.subtract_decay()
    experiment.da_fourier_transform()

    shutil.copytree(cfg['reports_dir'] / expt_date, cfg['vault_dir'] / expt_date / 'report', dirs_exist_ok=True)

    generate_dataset_notes(
        processed_data_dir=experiment.save_dir,
        vault_expt_dir=cfg['vault_dir'] / expt_date,
        expt_name=expt_date,
    )


# # %%

# %%
