"""Functions extracted from criteo.ipynb."""

import functools
import operator
import os
from pathlib import Path

import polars as pl
import xgboost as xgb
import yaml
from polars import selectors as cs
from wandb.integration.xgboost import WandbCallback

import wandb

N_INTS = 13
INTS = [f"I{i}" for i in range(1, N_INTS + 1)]
N_CATS = 26
CAT_COLS = [f"C{i}" for i in range(1, N_CATS + 1)]


def get_data_dir():
    path = Path("~/projects/data/Criteo").expanduser()
    if not path.exists():
        path = Path("/workspace/data/Criteo")
    return path


def create_ctr(df, cols):
    """Compute per-group click-through rate (CTR) with its deviation from the global CTR.

    Groups `df` by `cols` and, for each group, computes count, clicks and CTR, plus
    `diff` (group CTR minus global CTR), a 95%-confidence `margin` around the global
    CTR sized to the group's count, and `delta` (diff expressed in margins).

    Args:
        df: LazyFrame containing `cols` and a binary `Label` column.
        cols: column(s) to group by (a name, or a list of names).

    Returns:
        LazyFrame with one row per group, sorted by `count` descending.
    """
    ctr = df.select(pl.col("Label").mean()).collect().item(0, 0)
    df_ctr = (
        df.group_by(cols)
        .agg(
            pl.col("Label").len().alias("count"),
            pl.col("Label").sum().alias("clicks"),
            pl.col("Label").mean().alias("ctr"),
        )
        .with_columns(
            diff=pl.col("ctr") - pl.lit(ctr),
            margin=1.96 * (pl.lit(ctr) * (1 - pl.lit(ctr)) / pl.col("count")).sqrt(),
        )
        .with_columns(
            margin_n=-pl.col("margin"), delta=pl.col("diff").abs() / pl.col("margin")
        )
        .sort("count", descending=True)
    )
    return df_ctr


def get_data(config, data_dir):
    """load data from data_dir, sampling every config['sample_every']
    add missing category
    encode categoricals using target encoding
    split into X and y
    """
    train, valid, test = load_data(config["sample_every"], data_dir)
    datasets = [train, valid, test]
    for i_dataset, dataset in enumerate(datasets):
        datasets[i_dataset] = fill_missing(dataset, CAT_COLS, "missing")
    cat_encode, _gm = calc_cat_encode(datasets[0], config, CAT_COLS)

    if config.get("cat_order") == "target":
        for i_dataset, dataset in enumerate(datasets):
            datasets[i_dataset] = encode_cats(dataset, cat_encode, CAT_COLS)
    else:
        enums = create_enums(cat_encode)
        for i_dataset, dataset in enumerate(datasets):
            datasets[i_dataset] = cast_enums(merge_rare(dataset, cat_encode), enums)
    for i_dataset, dataset in enumerate(datasets):
        datasets[i_dataset] = get_data_XY(dataset)
    train, valid, test = datasets
    return train, valid, test


def load_data(sample_every, data_dir):
    """Lazily load the Criteo train/valid/test parquet files.

    The train set is subsampled to every `sample_every`-th row (by row index);
    valid and test are loaded in full.

    Args:
        sample_every: keep one row out of every `sample_every` rows of train.
        data_dir: directory containing train.parquet/valid.parquet/test.parquet.

    Returns:
        (train, valid, test) LazyFrames.
    """
    train = pl.scan_parquet(data_dir / "train.parquet", row_index_name="Row").filter(
        pl.col("Row").mod(sample_every) == 0
    )
    valid = pl.scan_parquet(data_dir / "valid.parquet")
    test = pl.scan_parquet(data_dir / "test.parquet")
    return train, valid, test


def fill_missing(df, cols, missing_val="missing"):
    """Replace nulls in `cols` with `missing_val`.

    Args:
        df: (Lazy)Frame containing `cols`.
        cols: column(s) to fill.
        missing_val: value to substitute for nulls.

    Returns:
        A copy of `df` with nulls in `cols` filled.
    """
    df = df.with_columns(pl.col(cols).fill_null(missing_val))
    return df


def calc_cat_encode(df, config, cat_cols):
    """Fit a smoothed target-CTR encoding for each categorical column in `cat_cols`.

    Calls `transform_cat` per column with `config['cat_cutoff']` and
    `config['cat_pseudo_count']`, and stacks the per-category rows with the pooled
    "rest" row into one table per column.

    Args:
        df: LazyFrame containing categorical columns and a `Label` column.
        config: dict with `cat_cutoff` and `cat_pseudo_count`.
        cat_cols: list of categorical column names to encode.

    Returns:
        (cat_encode, global_mean): cat_encode maps column name to its encoding
        DataFrame; global_mean is the overall Label mean (from the last column fit).
    """
    cat_encode = {}
    gm = None
    for col in cat_cols:
        cat_ctrs, rest_ctr, gm = calc_cat_smooth_ctr(
            df,
            col,
            "Label",
            cutoff=config["cat_cutoff"],
            pseudo_count=config["cat_pseudo_count"],
        )
        cat_encode[col] = pl.concat(
            (cat_ctrs.collect(), rest_ctr.collect()), how="diagonal"
        )
    return cat_encode, gm


def calc_cat_smooth_ctr(df, cols, label_col, cutoff, pseudo_count):
    """Target-encode a categorical column by its smoothed click-through rate (CTR).

    Groups `df` by `cols` and computes the count, clicks and raw CTR of each category.
    Categories with fewer than `cutoff` rows are pooled into a single "rest" row.
    Remaining categories get a smoothed CTR that is shrunk towards the global CTR:
        ctr_smooth = (clicks + global_mean * pseudo_count) / (count + pseudo_count)
    so low-count categories stay close to the global mean.

    Args:
        df: LazyFrame containing `cols` and `label_col`.
        cols: categorical column to group by (a name, or a list of names).
        label_col: binary click label column (0/1).
        cutoff: minimum row count for a category to be kept on its own.
        pseudo_count: prior strength; the number of rows of global-mean CTR blended in.

    Returns:
        df_ctr: LazyFrame with one row per kept category and columns `cols`, `count`,
            `clicks`, `ctr` and `ctr_smooth_diff` (ctr_smooth - global_mean, the encoding
            value), sorted by `ctr_smooth_diff` descending.
        rest: one-row LazyFrame for the pooled rare categories with `count`, `clicks`,
            `ctr` and `ctr_smooth` (absolute, not a difference from the global mean).
        global_mean: overall mean of `label_col` (float).
    """

    df_ctr = df.group_by(cols).agg(
        pl.col(label_col).len().alias("count"),
        pl.col(label_col).sum().alias("clicks"),
        pl.col(label_col).mean().alias("ctr"),
    )
    rest = df_ctr.filter(pl.col("count") < cutoff).select(["count", "clicks"]).sum()
    # rest is now row vector
    rest = rest.with_columns(ctr=pl.col("clicks") / pl.col("count"))

    df_ctr = df_ctr.filter(pl.col("count") >= cutoff)

    global_mean = df.select(label_col).mean().collect().item(0, 0)
    ctr_smooth = (pl.col("clicks") + global_mean * pseudo_count) / (
        pl.col("count") + pseudo_count
    )
    df_ctr = df_ctr.with_columns(ctr_smooth_diff=ctr_smooth - pl.lit(global_mean)).sort(
        "ctr_smooth_diff", descending=True
    )
    rest = rest.with_columns(ctr_smooth=ctr_smooth)
    return df_ctr, rest, global_mean


def merge_rare(df, cat_encode):
    for cat, cat_df in cat_encode.items():
        df = df.with_columns(
            pl.when(pl.col(cat).is_in(cat_df[cat].to_list()))
            .then(pl.col(cat))
            .otherwise(pl.lit("rare"))
        )
    return df


def create_enums(cat_encode):
    enums = {}
    for cat, cat_df in cat_encode.items():
        cat_df = cat_df.drop_nulls(cat).sort("count", descending=True)
        enums[cat] = pl.Enum(
            cat_df.get_column(cat).to_list() + ["rare"]
        )  # missing already there
    return enums


def cast_enums(df, enums):
    df = df.with_columns([pl.col(cat).cast(enum) for cat, enum in enums.items()])
    return df


def encode_cats(df, cat_encode, cat_cols):
    """Replace each categorical column in `cat_cols` with its target-encoded value.

    Left-joins each column in `cat_cols` against its encoding table from `cat_encode`
    (as produced by `calc_cat_encode`), drops the original column, and fills any
    unmatched (missing) encodings with 0.

    Args:
        df: LazyFrame containing `cat_cols` columns.
        cat_encode: dict mapping column name to its encoding DataFrame, keyed by
            `col` with a `{col}_encode` value column.
        cat_cols: list of categorical column names to encode.

    Returns:
        LazyFrame with each `cat` column replaced by a `{cat}_encode` column.
    """
    for cat in cat_cols:
        df = df.join(
            cat_encode[cat]
            .lazy()
            .rename({"ctr_smooth_diff": f"{cat}_encode"})
            .select([cat, f"{cat}_encode"]),
            how="left",
            on=cat,
        ).drop(cat)
    df = df.with_columns(cs.ends_with("_encode").fill_null(0)).rename(
        {f"{cat}_encode": cat for cat in cat_cols}
    )
    return df


def build_categorical_enums(
    paths: list[Path],
    cat_cols: list[str] = CAT_COLS,
) -> dict[str, pl.Enum]:
    """Build a polars Enum dtype per categorical column from its distinct values.

    Scans each parquet file in `paths` (concatenated) for each column in `cat_cols`
    and collects its sorted distinct non-null values into a `pl.Enum`.

    Args:
        paths: parquet files to scan (e.g. train/valid/test).
        cat_cols: categorical column names to build enums for.

    Returns:
        Dict mapping column name to its `pl.Enum` dtype.
    """
    enums: dict[str, pl.Enum] = {}
    for col in cat_cols:
        values = (
            pl.concat([pl.scan_parquet(p).select(col) for p in paths])
            .select(pl.col(col).drop_nulls().unique().sort())
            .collect(engine="streaming")
            .get_column(col)
        )
        enums[col] = pl.Enum(values)
    return enums


def get_data_XY(data):
    """prepare training data, select columns starting with I, ending with_encode or starting with ar_"""
    data_X = data.drop(["Row", "Label"], strict=False).collect()
    data_y = data.select("Label").collect().to_series()
    return data_X, data_y


def create_expression(pairs):
    """Build a polars filter expression matching all (column, value) pairs.

    Args:
        pairs: iterable of (col, val) tuples.

    Returns:
        A polars expression equivalent to `(col1 == val1) & (col2 == val2) & ...`.
    """
    return functools.reduce(
        operator.and_,
        (pl.col(col) == val for col, val in pairs),
    )


def parse_frequent(filename):
    """Parse a frequent-itemset text file into a polars LazyFrame.

    Each line is expected to end with a "(support%)" suffix and otherwise list
    "col:value" items separated by spaces. Adds a `rules` column holding a polars
    filter expression (via `create_expression`) for each itemset.

    Args:
        filename: path to the frequent-itemsets text file.

    Returns:
        LazyFrame with `text` (list of [col, value] pairs), `support` and `rules`.
    """
    df = (
        pl.scan_csv(filename, has_header=False, schema={"text": pl.String})
        .select(pl.col("text").str.split(" "))
        .with_columns(
            support=pl.col("text").list[-1].str.strip_chars("()").cast(pl.Float64)
            / 100,
            text=(
                pl.col("text")
                .list.head(pl.col("text").list.len() - 1)
                .list.eval(pl.element().str.split(":"))
                # .list.to_struct(upper_bound=5)
            ),
        )
        .with_columns(rules=pl.col("text").map_elements(create_expression, pl.Object))
    )
    return df


def parse_association(filename):
    """Parse an association-rule text file (rules predicting Label:1) into a LazyFrame.

    Each line is expected in the form "Label:1 <- item1 item2 ... (support, confidence)".
    Adds a `rules` column holding a polars filter expression (via `create_expression`)
    for each rule's antecedent items.

    Args:
        filename: path to the association-rules text file.

    Returns:
        LazyFrame with `items` (list of [col, value] pairs), `support`, `confidence`
        and `rules`.
    """
    # Label1: <- X X X (Y, Y)
    regex = (
        r"Label:1 <- (?P<items>[^(]+) \((?P<support>[0-9.]+), (?P<confidence>[0-9.]+)\)"
    )
    df = (
        pl.scan_csv(
            filename, has_header=False, schema={"text": pl.String}, separator="|"
        )
        .select(pl.col("text").str.extract_groups(regex))
        .unnest("text")
        .with_columns(pl.col("items").str.split(" "))
        .with_columns(
            pl.col(["support", "confidence"]).cast(pl.Float64) / 100,
            pl.col("items").list.eval(pl.element().str.split(":")),
        )
        .with_columns(rules=pl.col("items").map_elements(create_expression, pl.Object))
    )
    return df


def train():
    """wandb sweep agent entry point: trains an XGBClassifier for one sweep run."""
    config = {
        "sample_every": 1,
        # max_cat_to_onehot":,
        "n_estimators": 150,
        "learning_rate": 0.1,
        "gamma": 10,
        "min_child_weight": 1,
        "max_depth": 60,
        "device": "gpu",
        "tree_method": "hist",
        "grow_policy": "lossguide",  # depthwise or lossguide
        "early_stopping_rounds": 10,
        "max_bin": 256,
        "subsample": 1.0,
    }

    config["cat_order"] = "tree"
    config["cat_cutoff"] = 100
    config["reg_lambda"] = 10
    config["cat_pseudo_count"] = 16  # equivalent to cat_reg_lambda

    config["enable_categorical"] = config["cat_order"] in ["node", "tree", "ohe"]

    if config["cat_order"] in ["node", "tree"]:
        config["cat_reg_lambda"] = 400
        config["max_cat_threshold"] = 1_000_000
        config["max_cat_to_onehot"] = 10
    elif config["cat_order"] == "target":
        pass
    elif config["cat_order"] == "ohe":
        config["max_cat_threshold"] = 64
        config["max_cat_to_onehot"] = 4_294_967_295

    not_xgb_keys = ["sample_every", "cat_cutoff", "cat_pseudo_count"]
    if config["cat_order"] not in ["node", "tree"]:
        not_xgb_keys.append("cat_order")
    with wandb.init(config=config) as _run:
        config = wandb.config  # override with sweeps settings
        config_xgb = {
            key: value for key, value in config.items() if key not in not_xgb_keys
        }
        data_dir = get_data_dir()

        (train_X, train_y), (valid_X, valid_y), (_test_X, _test_y) = get_data(
            config, data_dir
        )
        clf = xgb.XGBClassifier(
            **config_xgb, callbacks=[WandbCallback(log_feature_importance=True)]
        )
        clf.fit(
            train_X,
            train_y,
            eval_set=[(train_X, train_y), (valid_X, valid_y)],
            verbose=2,
        )


if __name__ == "__main__":
    # All settings come from the environment:
    #   WANDB_PROJECT      - W&B project name
    #   WANDB_AGENT_RUNS   - number of runs this agent executes
    #   WANDB_SWEEP_CONFIG - sweep configuration as a (multiline) YAML string
    project = os.environ["WANDB_PROJECT"]
    agent_count = int(os.environ["WANDB_AGENT_RUNS"])
    sweep_configuration = yaml.safe_load(os.environ["WANDB_SWEEP_CONFIG"])
    if not isinstance(sweep_configuration, dict):
        raise ValueError("WANDB_SWEEP_CONFIG must be a YAML mapping")

    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # Start the sweep job
    wandb.agent(sweep_id, function=train, count=agent_count)
