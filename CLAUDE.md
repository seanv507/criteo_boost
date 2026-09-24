# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An experimentation repo for training an XGBoost click-through-rate (CTR) model on the
Criteo display-advertising dataset, plus supporting analysis of frequent itemsets /
association rules and of the resulting trees. Most of the actual work happens in the
`criteo.ipynb` notebook; `criteo_functions.py` holds the reusable, unit-tested pieces
extracted from it.

## Commands

Dependency management is via `uv` (Python `3.14`, pinned in `.python-version`;
`pyproject.toml` + `uv.lock` are the source of truth — don't `pip install` directly).

```bash
uv sync                 # install/update the environment
uv run jupyter lab      # open the notebook (criteo.ipynb)
uv run pytest           # run the test suite
uv run pytest tests/test_criteo_functions.py::test_transform_cat_smoothing_and_cutoff -q  # single test
```

To build the custom GPU-enabled XGBoost fork used for some experiments
(`seanv507/xgboost`), see `xgboost_build.md` (clone recursively, `uv pip install -e .`
in `python-package/`, then `cmake -GNinja` + `ninja` for the C++ side).

## Architecture

**`criteo_functions.py`** vs **`criteo.ipynb`**: functions were pulled out of the
notebook into this module (see git history) so they're importable and testable; the
notebook does `from criteo_functions import *` rather than redefining them inline. When
editing shared logic, change it in `criteo_functions.py`, not by re-adding a copy in the
notebook.

**Data pipeline** (`load_data` → `fill_missing` → `calc_cat_encode`/`transform_cat` →
`encode_cats` → `get_data_XY`, orchestrated by `get_data`): Criteo train/valid/test
parquet files have integer features `I1`-`I13`, categorical features `C1`-`C26`, and a
binary `Label`. Categoricals are not one-hot/enum encoded for the model — they're
target-encoded via a smoothed CTR (`transform_cat`: rare categories below a count
cutoff are pooled into one "rest" bucket; kept categories get
`ctr_smooth = (clicks + global_mean * pseudo_count) / (count + pseudo_count)`, shrinking
low-count categories toward the global mean). The final model matrix keeps only
`I*`, `*_encode`, and `ar_*` (association-rule) columns.

**Data location is external and inconsistently hardcoded** across the codebase —
`load_data()` in `criteo_functions.py` reads from `../data/Criteo` (relative to CWD),
while notebook cells elsewhere use `~/projects/data/Criteo` or `/workspace/data/Criteo`.
There's no config for this; when adding code that loads data, check which convention
the surrounding cell/function already uses rather than assuming one path.

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
run via `wandb.sweep(...)` + `wandb.agent(sweep_id, train)`, where `train()` in
`criteo_functions.py` is the sweep entry point (reads hyperparams from `wandb.config`,
fits an `XGBClassifier` with `WandbCallback`). `get_project_sweep_ids()` queries the
wandb API for every sweep across all projects under that entity. `wandb_workspaces` is
used to build/publish Report pages summarizing sweep results.

## Testing notes

`tests/test_criteo_functions.py` covers every function in `criteo_functions.py` except
`get_project_sweep_ids` (needs a live wandb API call) and `train` (a wandb sweep-agent
entry point tied to global sweep config). Tests use small synthetic polars
frames/parquet files rather than the real Criteo dataset — `load_data` and `get_data`
are tested via `monkeypatch` (chdir for `load_data`'s hardcoded relative path;
`monkeypatch.setattr(cf, "load_data", ...)` for `get_data`) so the suite never touches
the external data directory.
