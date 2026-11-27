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
  `criteo_functions`). Ends by calling `stop_self()` (from `criteo_functions`), which
  stops the current RunPod pod via its REST API (needs `RUNPOD_POD_ID`/`RUNPOD_API_KEY`).
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
If the optional `NTFY_TOPIC` is set, `notify()` pushes an ntfy.sh message when the
agent finishes or crashes (the free W&B plan has no alerts); `notify()` never raises.

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

`train.parquet`/`valid.parquet`/`test.parquet` are mirrored at
[huggingface.co/datasets/seanv507/Criteo](https://huggingface.co/datasets/seanv507/Criteo)
(download with the `hf` CLI, e.g. `hf download seanv507/Criteo test.parquet valid.parquet
train.parquet --repo-type dataset --local-dir <data_dir>`; see
`.agents/skills/hf-cli/SKILL.md`). The repo is a single-commit upload under the user's
own account with no dataset card, so it carries no provenance metadata of its own — but
`valid.parquet`/`test.parquet` each have exactly 4,584,062 rows with an identical click
count (1,174,544, CTR≈0.2562) and zero row overlap between them, matching the row counts
of the well-known "Criteo_x4" 8:1:1 benchmark split (train 36,672,493 / valid 4,584,062 /
test 4,584,062) used in CTR-prediction benchmark papers (FuxiCTR/BARS), not the raw
Criteo Labs release — consistent with the `Criteo_x4_001*` directory names already
present under the local data dir. Columns are `Label` (binary), `I1`-`I13` (integer,
with nulls), `C1`-`C26` (hashed hex-string categoricals, with nulls).

**Frequent itemsets / association rules**: mined outside Python using the `eclat` CLI
(not part of this repo/environment by default). The notebook exports categorical
columns as `Label:x`/`Ci:value` token lines (`*_items.tsv`) for `eclat` to consume, and
`eclat`'s output (`*_associations.txt`) is parsed back into polars via
`parse_frequent`/`parse_association`, which build a `rules` column of polars filter
expressions per itemset/rule using `create_expression`.

**Tree analysis** (`tree_analysis.py`) reconstructs, per node of a fitted XGBoost
booster, the summed gradient/hessian (`G`/`H`) bottom-up, then derives an independent
`Metric`/`Gain_Expected` per node. It's essentially a from-scratch reimplementation of
XGBoost's internal gain calculation, used to sanity-check/explore it rather than for
training. The file has two sections:

- **Fast path: numpy + pandas only, returns pandas.**
  - `tree_node_stats(booster, rounds_per_chunk=5)` gives one row per node.
  - `add_gradient_statistics(stats, lam, eta)` adds
    `H/G/Metric/Metric_Yes/Metric_No/Gain_Expected` vectorised, bottom-up one depth level
    at a time. `Metric_*` and `Gain_Expected` are NaN on leaves.
  - `get_tree_statistics_fast(model_or_booster, rounds_per_chunk=5)` wraps the two.

  This section deliberately avoids polars, despite the repo-wide polars preference: the
  user may submit it as a PR to upstream XGBoost, which uses pandas. Keep it free of
  polars and of this repo's other modules. It's about 6× faster than the reference on a
  500k-node model and scales to depth-60 models. Its columns use `tree_node_stats` names
  (`tree`, `node`, `left`, `right`, `cover`, …), not `trees_to_dataframe`'s (`Tree`,
  `Node`, `Yes`, `No`, `Cover`, …). Values agree with the reference to about 1e-3
  relative at worst: it reads float32 node arrays, and summing many leaves amplifies
  tiny differences where `G` is near 0.
- **Reference: polars, `get_tree_statistics(model)` and `add_level(df)`.** These go
  through `trees_to_dataframe()` and a per-tree Python loop, so they're only practical
  on small models. They return polars, use `trees_to_dataframe` column names, and are
  what the tests check the fast path against.

**Tree statistics for hyperparameter tuning** (`tree_analysis.tree_node_stats`, used in
`tree_metrics_analysis.ipynb`, a work in progress). The goal is to choose `gamma`,
`min_child_weight`, `max_depth` and so on from per-node statistics of a fitted model
instead of by trial and error. `tree_node_stats` came from a Claude session in
`~/projects/xgboost` (the user's XGBoost fork) as a separate `tree_stats.py`, later
merged into `tree_analysis.py`. What that session established:

- **Why `trees_to_dataframe()` is slow here.** With `max_depth=60` (the `train()`
  default), depth no longer limits the trees, so the node count grows with the number of
  training rows. A synthetic 1M-row test gave about 650k nodes per tree; 150 trees would
  be about 100M nodes. `trees_to_dataframe()` builds Python objects per node, about
  2–4 µs and about 900 B each, so roughly 90 GB at peak, which swaps. That explains the
  30+ minute runs. `get_tree_statistics` builds on `trees_to_dataframe()` and loops in
  Python per tree, so it shares the problem: only use it on small models. A saved `.ubj`
  model is about 35 B per node, so the file size gives the node count up front.
- **What `tree_node_stats` does differently.** It slices the booster
  (`booster[a:b]`, `rounds_per_chunk` rounds at a time) and reads the raw JSON node
  arrays with numpy. It computes depth breadth-first for all trees at once and uses
  compact dtypes (about 42 B per node). Columns: `tree, node, depth, is_leaf, left,
  right, missing` (child node ids within the tree, −1 on leaves; `missing` is the
  default-direction child, from the JSON `default_left` flag), `feature` (an index, not
  a name), `split, gain, cover, leaf_value`. On the depth-60 test it ran
  in 2.7 s against 8.6 s for `trees_to_dataframe`, with identical results. If even that
  is too much memory, reduce each chunk to per-depth summaries instead of keeping rows.
- **Categorical splits (checked in this repo, xgboost 3.3.0).** `gain`, `cover` and
  `leaf_value` match `trees_to_dataframe()`. On categorical split nodes, though,
  `split` holds junk (about `1e-45`) and the category set isn't extracted, so filter
  those nodes out before using `split`.
- **Reading the numbers.** The stored gain (`loss_changes`) has no ½ factor and no
  −γ term, unlike the textbook formula in the notebook's markdown cell. XGBoost prunes a
  split when `loss_chg < min_split_loss`, so `gain` is on the same scale as `gamma`
  directly. (`get_tree_statistics`' `Gain_Expected` follows the same convention.)
  `cover` is the hessian sum, so compare it with `min_child_weight`.
- **Survivorship bias.** Pruned splits are not stored, so only collect statistics from
  a permissive baseline (`gamma=0`, a small `min_child_weight`, a generous
  `max_depth`). Gains of candidate splits that were evaluated but rejected, or stopped
  by `max_depth`/`max_leaves`, would need C++ instrumentation in
  `src/tree/hist/evaluate_splits.h` in the fork.

**Experiment tracking**: Weights & Biases, entity `seanv507`. Hyperparameter sweeps are
run via `wandb.sweep(...)` + `wandb.agent(sweep_id, train)`, either interactively in
`criteo_train.ipynb` or headless via `python criteo_functions.py` (see Commands).
`train()` (in `criteo_functions.py`) is the sweep entry point: it builds a base config
(GPU `hist`, `cat_order="tree"`, etc.), lets `wandb.config` override it with the sweep's
hyperparameters, strips non-XGBoost keys (`sample_every`, `cat_cutoff`,
`cat_pseudo_count`, `inputs`, and `cat_order` unless it's `node`/`tree`), restricts
the feature matrices to `config["inputs"]` via `select_inputs` (comma-separated
tokens: `ints`, `cats`, `all`, or exact column names, e.g. `"ints,C1,C5"`; default
`"all"`), and fits an `XGBClassifier` with `WandbCallback`. `wandb_report.ipynb`'s
`get_project_sweep_ids()` queries the wandb API for every sweep across all projects
under that entity, and `wandb_workspaces` is used there to build/publish Report pages
summarizing sweep results.

## Testing notes

`tests/test_criteo_functions.py` and `tests/test_tree_analysis.py` cover every function
in `criteo_functions.py` and `tree_analysis.py` except `train()`, the `__main__` sweep
script, and `get_project_sweep_ids` (live wandb calls), which are untested by design.
The modules are importable from tests via `pythonpath = ["."]` under
`[tool.pytest.ini_options]` in `pyproject.toml`; they're top-level files, not a package.

Tests use small synthetic polars frames/parquet files rather than the real Criteo
dataset, so the suite never touches the external data directory. `get_data` is tested
with `monkeypatch.setattr(cf, "load_data", ...)`, once for each encoding path
(`cat_order="target"` and an enum path). `get_data_dir` is tested by pointing `HOME` at
a tmp dir.
