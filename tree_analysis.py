import json

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from tree_stats import tree_node_stats


def _lambda_eta(booster):
    params = json.loads(booster.save_config())["learner"]["gradient_booster"][
        "tree_train_param"
    ]
    return float(params["lambda"]), float(params["eta"])


def add_gradient_statistics(stats: pd.DataFrame, lam: float, eta: float):
    """Add get_tree_statistics' G/H/Metric/Gain_Expected columns to tree_node_stats output.

    Vectorised equivalent of the per-tree loop in `get_tree_statistics`: leaf `G` is
    recovered from the leaf value (`-leaf_value * (H + lam) / eta`), then internal
    nodes are filled bottom-up one depth level at a time as `G[left] + G[right]`.

    Args:
        stats: output of `tree_stats.tree_node_stats` (needs tree, node, depth,
            is_leaf, left, right, cover, leaf_value).
        lam: the model's `lambda` (reg_lambda).
        eta: the model's learning rate.

    Returns:
        polars DataFrame of `stats` plus `H`, `G`, `Metric` (= G^2 / (H + lam)),
        `Metric_Yes`, `Metric_No` (the children's Metric) and `Gain_Expected`
        (= Metric_Yes + Metric_No - Metric). The last three are null on leaves.
    """
    tree = stats["tree"].to_numpy()
    node = stats["node"].to_numpy().astype(np.int64)
    depth = stats["depth"].to_numpy()
    is_leaf = stats["is_leaf"].to_numpy()
    internal = ~is_leaf

    # Map (tree, node) -> row. Node ids aren't contiguous (unreachable nodes were
    # dropped), so give each tree a slot range of size max(node) + 1.
    n_slots = np.zeros(tree.max() + 1, dtype=np.int64)
    np.maximum.at(n_slots, tree, node + 1)
    base = np.concatenate([[0], np.cumsum(n_slots)[:-1]])
    row_of = np.full(n_slots.sum(), -1, dtype=np.int64)
    row_of[base[tree] + node] = np.arange(len(stats))
    left_row = np.where(internal, row_of[base[tree] + stats["left"].to_numpy()], -1)
    right_row = np.where(internal, row_of[base[tree] + stats["right"].to_numpy()], -1)

    H = stats["cover"].to_numpy().astype(np.float64)
    G = np.where(
        is_leaf,
        -stats["leaf_value"].to_numpy().astype(np.float64) * (H + lam) / eta,
        0.0,
    )
    # Internal rows grouped by depth, deepest first: children are always done first.
    rows = np.flatnonzero(internal)
    rows = rows[np.argsort(depth[rows], kind="stable")]
    bounds = np.flatnonzero(np.diff(depth[rows])) + 1
    for level in reversed(np.split(rows, bounds)):
        G[level] = G[left_row[level]] + G[right_row[level]]

    metric = G**2 / (H + lam)
    metric_yes = np.where(internal, metric[left_row], np.nan)
    metric_no = np.where(internal, metric[right_row], np.nan)
    return pl.from_pandas(
        stats.assign(
            H=H,
            G=G,
            Metric=metric,
            Metric_Yes=metric_yes,
            Metric_No=metric_no,
            Gain_Expected=metric_yes + metric_no - metric,
        )
    )


def get_tree_statistics_fast(model, rounds_per_chunk: int = 5):
    """`get_tree_statistics` built on `tree_node_stats` instead of `trees_to_dataframe`.

    Same G/H/Metric/Gain_Expected values, but scales to large (e.g. depth-60) models.
    Columns follow tree_node_stats naming (tree, node, left, right, cover, ...)
    rather than trees_to_dataframe's (Tree, Node, Yes, No, Cover, ...).

    Args:
        model: a fitted XGBoost sklearn model, or a `xgb.Booster`.
        rounds_per_chunk: passed to `tree_node_stats`; lower it if memory is tight.
    """
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    lam, eta = _lambda_eta(booster)
    return add_gradient_statistics(tree_node_stats(booster, rounds_per_chunk), lam, eta)


def get_tree_statistics(model: xgb.XGBModel):
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

    lam, eta = _lambda_eta(booster)
    df = df.with_columns(H=pl.col("Cover"))  # hessian sum, valid for every row
    is_leaf = pl.col("Feature") == "Leaf"
    df = df.with_columns(
        G=pl.when(is_leaf).then(-pl.col("Gain") * (pl.col("H") + lam) / eta)
    )
    tdfs = {}
    for tree_id, tdf in df.sort("Node", descending=True).group_by("Tree"):
        g = dict(zip(tdf["ID"], tdf["G"]))

        intermediate_node = tdf.filter(~is_leaf)
        for row in intermediate_node.iter_rows(named=True):
            g[row["ID"]] = (
                g[row["Yes"]] + g[row["No"]]
            )  # missing already folds into Yes/No
        tdf = tdf.with_columns(G=pl.col("ID").replace_strict(g)).with_columns(
            Metric=pl.col("G") ** 2 / (pl.col("H") + lam)
        )
        metric = dict(zip(tdf["ID"], tdf["Metric"]))
        tdf = tdf.with_columns(
            pl.col(["Yes", "No"]).replace_strict(metric).name.prefix("Metric_")
        ).with_columns(
            Gain_Expected=pl.col("Metric_Yes") + pl.col("Metric_No") - pl.col("Metric")
        )

        tdfs[tree_id] = tdf
    df = pl.concat(tdfs.values())
    return df


def add_level(df):
    all_nodes = []
    level = 0
    trees = df
    trees = trees.with_columns(is_parent=pl.col("Node") == 0)
    while len(trees) > 0:
        trees_parent_child = trees.partition_by("is_parent", as_dict=True)
        trees_parent = trees_parent_child[(True,)]
        nodes = trees_parent.select(
            ["ID", "Tree", "Node", pl.lit(level).alias("Level")]
        )
        all_nodes.append(nodes)
        trees = trees_parent_child.get((False,))
        if trees is None:
            break
        level += 1
        child_nodes = (
            trees_parent.select(["Yes", "No"])
            .unpivot(variable_name="Branch", value_name="ID")
            .drop_nulls()
        )
        trees = trees.with_columns(
            is_parent=pl.col("ID").is_in(child_nodes["ID"].to_list())
        )
    all_nodes = pl.concat(all_nodes)
    df = df.join(all_nodes.select(["ID", "Level"]), on="ID")
    return df
