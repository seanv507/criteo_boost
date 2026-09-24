# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An experimentation repo for training an XGBoost click-through-rate (CTR) model on the
Criteo display-advertising dataset, plus supporting analysis of frequent itemsets /
association rules and of the resulting trees. `criteo_functions.py` holds the reusable,
unit-tested pieces; the work itself is split across three notebooks:

- `criteo_train.ipynb` — the minimal path to load/encode data and train a model (a
  single fit, a GPU variant, and a wandb hyperparameter sweep). Defines its own
  `train()` (the sweep-agent entry point) locally rather than importing it, since it's
  only ever used here.
- `criteo_analysis.ipynb` — everything else: exploratory analysis, frequent-itemset/
  association-rule mining, and tree-structure inspection. This one is a preserved
  scratchpad, not a clean pipeline — several cells are known-broken (undefined
  variables from deleted precursor cells, a missing import, a dead function call) and
  are flagged with a `⚠️` markdown note directly above them rather than fixed, so treat
  those notes as authoritative before trying to run a flagged cell.
- `wandb_report.ipynb` — queries wandb sweep results and builds a summary table +
  `wandb_workspaces` Report. Self-contained (doesn't import `criteo_functions`);
  defines its own local `get_project_sweep_ids()`. Also a preserved scratchpad with a
  couple of annotated broken/duplicate cells, same convention as `criteo_analysis.ipynb`.

## Commands

Dependency management is via `uv` (Python `3.14`, pinned in `.python-version`;
`pyproject.toml` + `uv.lock` are the source of truth — don't `pip install` directly).

```bash
uv sync                 # install/update the environment
uv run jupyter lab      # open the notebooks (criteo_train.ipynb / criteo_analysis.ipynb / wandb_report.ipynb)
uv run pytest           # run the test suite
uv run pytest tests/test_criteo_functions.py::test_transform_cat_smoothing_and_cutoff -q  # single test
```

To build the custom GPU-enabled XGBoost fork used for some experiments
(`seanv507/xgboost`), see `xgboost_build.md` (clone recursively, `uv pip install -e .`
in `python-package/`, then `cmake -GNinja` + `ninja` for the C++ side).

## Architecture

**`criteo_functions.py`** vs the notebooks: functions were pulled out of the original
notebook into this module (see git history) so they're importable, testable, and
shared between `criteo_train.ipynb` and `criteo_analysis.ipynb`, both of which do
`from criteo_functions import *` rather than redefining anything inline. When editing
shared logic, change it here, not by re-adding a copy in a notebook. Two exceptions
were deliberately moved back *out* of this module and into the one notebook that
actually uses them, rather than being shared: `train()` (now defined directly in
`criteo_train.ipynb`) and `get_project_sweep_ids()` (now defined directly in
`wandb_report.ipynb`) — neither is unit-tested for the same reason.

**Data pipeline** (`load_data` → `fill_missing` → `calc_cat_encode`/`transform_cat` →
`encode_cats` → `get_data_XY`, orchestrated by `get_data`): Criteo train/valid/test
parquet files have integer features `I1`-`I13`, categorical features `C1`-`C26`, and a
binary `Label`. Categoricals are not one-hot/enum encoded for the model — they're
target-encoded via a smoothed CTR (`transform_cat`: rare categories below a count
cutoff are pooled into one "rest" bucket; kept categories get
`ctr_smooth = (clicks + global_mean * pseudo_count) / (count + pseudo_count)`, shrinking
low-count categories toward the global mean). The final model matrix keeps only
`I*`, `*_encode`, and `ar_*` (association-rule) columns.

**Data location is external**, and not fully unified across the codebase. `load_data()`/
`get_data()` in `criteo_functions.py` take an explicit `data_dir` parameter (default
`../data/Criteo`, relative to CWD; `get_data` reads it from `config['data_dir']`), so
`criteo_train.ipynb` sets it once at the top. `criteo_analysis.ipynb`'s exploratory
cells, however, still use several different hardcoded literal paths inherited from the
original notebook (`~/projects/data/Criteo`, `/workspace/data/Criteo`, `../data/Criteo`)
in direct `pl.scan_parquet`/`pl.read_parquet` calls — check which convention the
surrounding cell uses rather than assuming one path.

**Frequent itemsets / association rules**: mined outside Python using the `eclat` CLI
(not part of this repo/environment by default). The notebook exports categorical
columns as `Label:x`/`Ci:value` token lines (`*_items.tsv`) for `eclat` to consume, and
`eclat`'s output (`*_associations.txt`) is parsed back into polars via
`parse_frequent`/`parse_association`, which build a `rules` column of polars filter
expressions per itemset/rule using `create_expression`.

**Tree analysis** (`get_tree_statistics`): reconstructs, per split node of a fitted
XGBoost booster, the summed gradient/hessian (`G`/`H`) bottom-up from
`trees_to_dataframe()`, then derives an independent `Metric`/`Gain_Expected` per node —
essentially a from-scratch reimplementation of XGBoost's internal gain calculation, used
to sanity-check/explore it rather than for training.

**Experiment tracking**: Weights & Biases, entity `seanv507`. Hyperparameter sweeps are
run via `wandb.sweep(...)` + `wandb.agent(sweep_id, train)` in `criteo_train.ipynb`,
where `train()` is the sweep entry point (reads hyperparams from `wandb.config`, fits
an `XGBClassifier` with `WandbCallback`). `wandb_report.ipynb`'s
`get_project_sweep_ids()` queries the wandb API for every sweep across all projects
under that entity, and `wandb_workspaces` is used there to build/publish Report pages
summarizing sweep results.

## Testing notes

`tests/test_criteo_functions.py` covers every function currently in
`criteo_functions.py`. `get_project_sweep_ids` (needs a live wandb API call) and
`train` (tied to global sweep config) are untested by design — and live in
`wandb_report.ipynb`/`criteo_train.ipynb` respectively rather than in this module, so
they aren't importable for testing anyway. Tests use small synthetic polars
frames/parquet files rather than the real Criteo dataset — `load_data` and `get_data`
are tested via `monkeypatch` (chdir for `load_data`'s hardcoded relative path;
`monkeypatch.setattr(cf, "load_data", ...)` for `get_data`) so the suite never touches
the external data directory.
