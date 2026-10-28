# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An experimentation repo for training an XGBoost click-through-rate (CTR) model on the
Criteo display-advertising dataset, plus supporting analysis of frequent itemsets /
association rules and of the resulting trees. `criteo_functions.py` holds the reusable
pieces (data pipeline, rule parsing, and the `train()` sweep entry point) and doubles as
a runnable wandb sweep-agent script; `tree_analysis.py` holds the tree-inspection
helpers. The interactive work is split across three notebooks:

- `criteo_train.ipynb` — the minimal path to load/encode data and train a model (a
  single fit, a GPU variant, and a wandb hyperparameter sweep using `train()` from
  `criteo_functions`). Ends with a `stop_self()` helper that stops the current RunPod
  pod via its REST API (needs `RUNPOD_POD_ID`/`RUNPOD_API_KEY`).
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
uv run pytest tests/test_criteo_functions.py::test_create_expression_single_pair -q  # single test

# run a wandb sweep agent headless (e.g. on a GPU pod); all three env vars are required
export WANDB_PROJECT=criteo_boost_gpu_categorical WANDB_AGENT_RUNS=10
export WANDB_SWEEP_CONFIG="$(cat sweep.yaml)"   # multiline YAML sweep config
uv run python criteo_functions.py
```

To build the custom GPU-enabled XGBoost fork used for some experiments
(`seanv507/xgboost`), see `xgboost_build.md` (clone recursively, `uv pip install -e .`
in `python-package/`, then `cmake -GNinja` + `ninja` for the C++ side).

## Architecture

**`criteo_functions.py`** vs the notebooks: functions were pulled out of the original
notebook into this module (see git history) so they're importable, testable, and
shared between `criteo_train.ipynb` and `criteo_analysis.ipynb`, both of which do
`from criteo_functions import *` rather than redefining anything inline. When editing
shared logic, change it here, not by re-adding a copy in a notebook. The one exception
is `get_project_sweep_ids()`, deliberately kept in `wandb_report.ipynb` (the only place
it's used). `train()` was briefly moved into `criteo_train.ipynb` but now lives back in
this module so the same entry point serves both the notebook and the headless
`__main__` sweep script.

The `__main__` block reads `WANDB_PROJECT` (project name), `WANDB_AGENT_RUNS` (int,
`count` for `wandb.agent`) and `WANDB_SWEEP_CONFIG` (a YAML mapping, parsed with
`yaml.safe_load`) from the environment — no defaults, a missing variable raises
`KeyError` — then calls `wandb.sweep` + `wandb.agent(sweep_id, train, count=...)`.

**Data pipeline** (`load_data` → `fill_missing` → `calc_cat_encode`/`calc_cat_smooth_ctr`
→ encode, then `get_data_XY`, orchestrated by `get_data(config, data_dir)`): Criteo
train/valid/test parquet files have integer features `I1`-`I13`, categorical features
`C1`-`C26`, and a binary `Label`. `calc_cat_smooth_ctr` fits a smoothed CTR per
category: categories below `config['cat_cutoff']` rows are pooled into one "rest"
bucket; kept categories get
`ctr_smooth = (clicks + global_mean * pseudo_count) / (count + pseudo_count)`, shrinking
low-count categories toward the global mean. How that's used depends on
`config['cat_order']`:
- `"target"` — `encode_cats` replaces each categorical with its smoothed-CTR diff
  (target encoding).
- anything else (`"node"`, `"tree"`, `"ohe"`, `None`) — `merge_rare` maps rare
  categories to `"rare"` and `create_enums`/`cast_enums` cast columns to `pl.Enum`, for
  XGBoost's native categorical support (`enable_categorical`).

**Data location is external**, and not fully unified across the codebase.
`load_data()`/`get_data()` take a required `data_dir` (`Path`); callers get it from
`get_data_dir()`, which returns `~/projects/data/Criteo` if it exists, else
`/workspace/data/Criteo` (the RunPod layout). `criteo_analysis.ipynb`'s exploratory
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

**Tree analysis** (`tree_analysis.py`: `get_tree_statistics`, `add_level`): reconstructs, per split node of a fitted
XGBoost booster, the summed gradient/hessian (`G`/`H`) bottom-up from
`trees_to_dataframe()`, then derives an independent `Metric`/`Gain_Expected` per node —
essentially a from-scratch reimplementation of XGBoost's internal gain calculation, used
to sanity-check/explore it rather than for training.

**Experiment tracking**: Weights & Biases, entity `seanv507`. Hyperparameter sweeps are
run via `wandb.sweep(...)` + `wandb.agent(sweep_id, train)`, either interactively in
`criteo_train.ipynb` or headless via `python criteo_functions.py` (see Commands).
`train()` (in `criteo_functions.py`) is the sweep entry point: it builds a base config
(GPU `hist`, `cat_order="tree"`, etc.), lets `wandb.config` override it with the sweep's
hyperparameters, strips non-XGBoost keys (`sample_every`, `cat_cutoff`,
`cat_pseudo_count`, and `cat_order` unless it's `node`/`tree`), and fits an
`XGBClassifier` with `WandbCallback`. `wandb_report.ipynb`'s
`get_project_sweep_ids()` queries the wandb API for every sweep across all projects
under that entity, and `wandb_workspaces` is used there to build/publish Report pages
summarizing sweep results.

## Testing notes

`tests/test_criteo_functions.py` uses small synthetic polars frames/parquet files rather
than the real Criteo dataset, and `monkeypatch` (`chdir`, `setattr(cf, "load_data",
...)`) so it never touches the external data directory. `train()`, the `__main__`
sweep script, and `get_project_sweep_ids` (live wandb calls) are untested by design.

**The suite is currently out of date and does not run:**
- `uv run pytest` fails at collection with `ModuleNotFoundError: No module named
  'criteo_functions'` (the repo root isn't on `sys.path`; there's no
  `conftest.py`/`pythonpath` setting).
- Even once importable, it targets an older API: it calls `cf.transform_cat` (now
  `calc_cat_smooth_ctr`) and `cf.get_tree_statistics` (now in `tree_analysis.py`), calls
  `load_data` without `data_dir`, and calls `get_data(config)` expecting six return
  values (it now takes `data_dir` and returns three `(X, y)` pairs).
- `merge_rare`, `create_enums`, `cast_enums` and `get_data_dir` have no tests.
