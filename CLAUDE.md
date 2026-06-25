# CLAUDE.md — TRR / Steady-State Analysis Package

Orientation for working in this repo. This is a Python package for processing and analyzing optical spectroscopy data (time-resolved reflectance / TRR, and steady-state PL including B-field-dependent measurements), integrated with an Obsidian lab notebook.

For current status and what's being worked on now, see `foundational-system-plan.md` — this file holds the durable context; the plan file holds the evolving task state.

## Architecture

- **`processing_package/` is an installable package** (`pip install -e .`, run in a venv). Projects *import* it — they never contain a copy of it. Improvements propagate to all projects automatically.
- **Package provides capability; the project provides context.** The package makes no assumption about where data lives. All paths and parameters come from a project-level `config.yaml` passed in — never hardcoded in package code. Preserve this strictly: any new function that needs a location takes it as an argument sourced from config.
- **Projects** are separate folders (one per experiment campaign), each with its own `config.yaml`. Data lives on **Box**; the Obsidian vault is **local** (synced via Obsidian Sync). Both are just local filesystem paths to the code.
- **`scripts/`** holds driver scripts — runnable orchestration that wires package capabilities together for an experiment. Drivers use config; they don't contain processing logic.

## Module organization

Read the file tree directly for what exists. What matters here is the *organizing principle* and where new code goes:

- *Shared operations* → operation-modules (`utilities` for filename parsing + small helpers; `fitting_functions` for fitting primitives; `fft_functions`, `noise_remover`).
- *Data operations on a dataset* → the `trrxr` xarray accessor in `trr_dataset`.
- *Workflow orchestration* → its own module (`processing_state` for the TRR pipeline).
- *Data-type-specific code* → that type's module (`pl_bfield`, `steady_state`).

Module-specific intent and rules live in each file's header docstring — read those before modifying a module.

**Placement test for new code:**
- *Imported and reused* → package. *Run to do a job* → `scripts/`.
- *Data operation* → accessor method. *File I/O or orchestration* → standalone function or driver.
- *Shared across data types* → operation module. *Specific to one* → that data-type's module.

## Conventions

- **Filename format:** `##-[sample]-[temp]K-[bfield]T-[scale]SF-[R0]mV-[extra tokens]`, dash-delimited, `p` as decimal point (e.g. `01-flake1a-1p6K-0p2T-70kSF-120mV-2pass-offset1`). `##` is a per-experiment dataset ID. Tags are identified by suffix/format, not position. Unrecognized tokens are captured as a freeform list, not dropped.
- **Config (`config.yaml`)** lives in each project folder; paths resolve relative to the config file (except absolute paths like the vault). Keys include `data_dir`, `processed_data_dir`, `reports_dir`, `pl_dir`, `vault_dir`.
- **Verification by oracle:** correctness is checked by reproducing known-good output, not by inspection. Golden reference files (e.g. `golden_trr.nc`) exist for this. After any refactor that could touch results, run the pipeline and compare with `xr.testing.assert_allclose` (use `allclose`, never exact equality — floating-point reordering is expected). Compare variables *by name*, and compare ground truth (the raw file) when in doubt, not just the prior version.

## Settled architecture decisions

- **Do NOT generalize `ProcessingState` to cover steady-state.** TRR is a linear step-pipeline with a per-step audit report; steady-state is interactive fitting + cross-dataset comparison. They are different shapes. Share *helpers*; keep *orchestration* separate. Forcing steady-state into the pipeline frame is the wrong abstraction.
- Quick-look plots for Obsidian reference the already-generated processed-step plot rather than regenerating.

## Working principles

- **Solid, not finished.** Build for what's actually needed now; let real data reveal what to refine. Don't build against imagined future needs.
- **Foundation vs. refinement.** Distinguish "needed to collect/analyze data" from "would be nicer." Don't promote the latter just because there's time.
- **Capture, don't act.** When noticing tangential cleanup or improvements, note them (in the plan file's cleanup/refinement sections) rather than doing them mid-task.
- **Scoped changes.** Do only what's asked. Do not refactor, "clean up," or restructure adjacent code as a side effect. If something else looks worth changing, flag it — don't do it.
- **Stage and verify.** Make one change at a time; verify (against the oracle where relevant) before the next. Don't assume a refactor preserved behavior — check.
- **Don't change processing math** when the task is structural. Inputs/outputs of calculations must stay identical unless changing them is the explicit goal.

## Collaboration preferences

- Concise responses. Think things through when it matters, but don't pad.
- Prefer making the decision *mine* on design questions — lay out real options and tradeoffs rather than picking silently.
- When given a tightly-scoped task, respect the scope boundary exactly.

## Environment notes

- Windows. Use `cmd` (Git Bash has a broken PATH — known issue, not yet fixed).
- Activate the venv before running code: `.venv\Scripts\activate.bat`.
- `.venv/`, `*.egg-info/`, `__pycache__/` are gitignored.
