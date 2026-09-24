"""Functions extracted from criteo.ipynb."""

import functools
import json
import operator
from pathlib import Path

import polars as pl
from polars import selectors as cs

CAT_COLS = [f"C{i}" for i in range(1, 27)]
data_dir = Path("../data/Criteo")
N_INTS = 13
INTS = [f"I{i}" for i in range(1, N_INTS + 1)]
N_CATS = 26
CAT_COLS = [f"C{i}" for i in range(1, N_CATS + 1)]

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
    ctr = df.select(pl.col('Label').mean()).collect().item(0, 0)
    df_ctr = (
        df
        .group_by(cols)
        .agg(
            pl.col('Label').len().alias('count'),
            pl.col('Label').sum().alias('clicks'),
            pl.col('Label').mean().alias('ctr'),
        )
        .with_columns(
            diff=pl.col('ctr') - pl.lit(ctr),
            margin=1.96 * (pl.lit(ctr) * (1 - pl.lit(ctr)) / pl.col('count')).sqrt()
        )
        .with_columns(
            margin_n=-pl.col('margin'),
            delta=pl.col('diff').abs() / pl.col('margin')
        )
        .sort('count', descending=True)
    )
    return df_ctr


def get_data(config):
    """ load data, sampling every config['sample_every']
        add missing category
        encode categoricals using target encoding
        split into X and y

        config['data_dir'] optionally overrides the default Criteo parquet directory.
    """
    data_dir = config.get('data_dir', Path("../data/Criteo"))
    train, valid, test = load_data(config['sample_every'], data_dir)

    train = fill_missing(train, CAT_COLS, "missing")
    valid = fill_missing(valid, CAT_COLS, "missing")
    test = fill_missing(test, CAT_COLS, "missing")
    cat_encode, gm = calc_cat_encode(train, config, CAT_COLS)
    train = encode_cats(train, cat_encode, CAT_COLS)
    valid = encode_cats(valid, cat_encode, CAT_COLS)
    test = encode_cats(test, cat_encode, CAT_COLS)

    train_X, train_y = get_data_XY(train)
    valid_X, valid_y = get_data_XY(valid)
    test_X, test_y = get_data_XY(test)
    return train_X, train_y, valid_X, valid_y, test_X, test_y


def load_data(sample_every, data_dir=Path("../data/Criteo")):
    """Lazily load the Criteo train/valid/test parquet files.

    The train set is subsampled to every `sample_every`-th row (by row index);
    valid and test are loaded in full.

    Args:
        sample_every: keep one row out of every `sample_every` rows of train.
        data_dir: directory containing train.parquet/valid.parquet/test.parquet.

    Returns:
        (train, valid, test) LazyFrames.
    """
    train = pl.scan_parquet(data_dir / "train.parquet", row_index_name='Row').filter(pl.col('Row').mod(sample_every) == 0)
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
    for col in cat_cols:
        cat_ctrs, rest_ctr, gm = transform_cat(df, col, 'Label', cutoff=config['cat_cutoff'], pseudo_count=config['cat_pseudo_count'])
        cat_encode[col] = pl.concat((cat_ctrs.collect(), rest_ctr.collect()), how='diagonal')
    return cat_encode, gm


def transform_cat(df, cols, label_col, cutoff, pseudo_count):
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

    df_ctr = (
        df
        .group_by(cols)
        .agg(
            pl.col(label_col).len().alias('count'),
            pl.col(label_col).sum().alias('clicks'),
            pl.col(label_col).mean().alias('ctr'),
        )
    )
    rest = df_ctr.filter(pl.col('count') < cutoff).select(['count', 'clicks']).sum()
    # rest is now row vector
    rest = rest.with_columns(ctr=pl.col('clicks') / pl.col('count'))

    df_ctr = df_ctr.filter(pl.col('count') >= cutoff)

    global_mean = df.select(label_col).mean().collect().item(0, 0)
    ctr_smooth = (pl.col('clicks') + global_mean * pseudo_count) / (pl.col('count') + pseudo_count)
    df_ctr = df_ctr.with_columns(
        ctr_smooth_diff=ctr_smooth - pl.lit(global_mean)
    ).sort('ctr_smooth_diff', descending=True)
    rest = rest.with_columns(ctr_smooth=ctr_smooth)
    return df_ctr, rest, global_mean


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
        df = df.join(cat_encode[cat].lazy().rename({'ctr_smooth_diff': f'{cat}_encode'}).select([cat, f'{cat}_encode']), how='left', on=cat).drop(cat)
    df = df.with_columns(cs.ends_with("_encode").fill_null(0))
    return df


def get_data_XY(data):
    """ prepare training data, select columns starting with I, ending with_encode or starting with ar_"""
    data_X = data.drop(['Row', 'Label'], strict=False).select(cs.starts_with("I") | cs.ends_with("_encode") | cs.starts_with("ar_")).collect()
    data_y = data.select('Label').collect().to_series()
    return data_X, data_y


def get_tree_statistics(model):
    """Compute per-node gradient/hessian stats and expected gain for an XGBoost model.

    Walks each tree bottom-up from `booster.trees_to_dataframe()` to reconstruct each
    node's summed gradient `G` and hessian `H` (leaves from Gain/Cover, internal nodes
    from their children), then derives `Metric` (= G^2 / (H + lambda)) and
    `Gain_Expected` (the metric gain from splitting vs. not) per node.

    Args:
        model: a fitted XGBoost sklearn model (e.g. XGBClassifier).

    Returns:
        A polars DataFrame with one row per tree node, including `G`, `H`, `Metric`
        and `Gain_Expected` columns, concatenated across all trees.
    """
    booster = model.get_booster()
    df = pl.from_pandas(booster.trees_to_dataframe())

    lam = float(json.loads(booster.save_config())["learner"]["gradient_booster"]["tree_train_param"]["lambda"])
    eta = float(json.loads(booster.save_config())["learner"]["gradient_booster"]["tree_train_param"]["eta"])
    df = df.with_columns(H=pl.col("Cover"))  # hessian sum, valid for every row
    is_leaf = pl.col("Feature") == "Leaf"
    df = df.with_columns(G=pl.when(is_leaf).then(-pl.col("Gain") * pl.col("H") + lam) / eta)
    tdfs = {}
    for tree_id, tdf in df.sort("Node", descending=True).group_by("Tree"):

        print(tree_id)
        g = dict(zip(tdf["ID"], tdf["G"]))

        intermediate_node = tdf.filter(~is_leaf)
        for row in intermediate_node.iter_rows(named=True):
            g[row["ID"]] = g[row["Yes"]] + g[row["No"]]  # missing already folds into Yes/No
        tdf = (
            tdf
            .with_columns(G=pl.col("ID").replace_strict(g))
            .with_columns(Metric=pl.col("G") ** 2 / (pl.col("H") + lam))

        )
        metric = dict(zip(tdf["ID"], tdf["Metric"]))
        tdf = (
            tdf
            .with_columns(pl.col(["Yes", "No"]).replace_strict(metric).name.prefix("Metric_"))
            .with_columns(Gain_Expected=pl.col("Metric_Yes") + pl.col("Metric_No") - pl.col("Metric"))
        )

        tdfs[tree_id] = tdf
    df = pl.concat(tdfs.values())
    return df


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
        pl.scan_csv(filename, has_header=False, schema={"text": pl.String}).select(
            pl.col('text')
            .str.split(" ")
        ).with_columns(
            support=pl.col("text").list[-1].str.strip_chars("()").cast(pl.Float64) / 100,
            text=(
                pl.col("text")
                .list.head(pl.col("text").list.len() - 1)
                .list.eval(pl.element().str.split(":"))
                # .list.to_struct(upper_bound=5)
            )
        ).with_columns(
            rules=pl.col('text').map_elements(create_expression, pl.Object)
        )
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
    regex = r"Label:1 <- (?P<items>[^(]+) \((?P<support>[0-9.]+), (?P<confidence>[0-9.]+)\)"
    df = (
        pl.scan_csv(filename, has_header=False, schema={"text": pl.String}, separator="|").select(
            pl.col('text')
            .str.extract_groups(regex)
        ).unnest('text')
        .with_columns(pl.col("items").str.split(" "))
        .with_columns(
            pl.col(["support", "confidence"]).cast(pl.Float64) / 100,
            pl.col('items')
            .list.eval(pl.element().str.split(":"))
        )
        .with_columns(
            rules=pl.col('items').map_elements(create_expression, pl.Object)
        )
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
