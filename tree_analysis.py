"""Per-node statistics for fitted XGBoost tree models.

The first section (`tree_node_stats`, `add_gradient_statistics`,
`get_tree_statistics_fast`) uses only numpy, pandas and the standard library and
returns pandas DataFrames. It is vectorised and processes the model a few boosting
rounds at a time (``booster[a:b]``), so peak memory is bounded by one chunk of trees
rather than by the whole model, and it scales to large (e.g. depth-60) models.

The second section (`get_tree_statistics`, `add_level`) is the original polars
implementation on top of `booster.trees_to_dataframe()`. It is only practical on
small models, and is kept as the reference the fast path is tested against.
"""

import json

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

# ---------------------------------------------------------------------------
# Fast path: numpy + pandas, chunked over boosting rounds
# ---------------------------------------------------------------------------


def _chunk(booster, tree_offset: int) -> pd.DataFrame:
    model = json.loads(booster.save_raw(raw_format="json"))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    del model
    sizes = np.array([len(t["left_children"]) for t in trees])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def cat(key, dtype):
        return np.concatenate([np.asarray(t[key], dtype=dtype) for t in trees])  # noqa: F821

    left = cat("left_children", np.int64)
    right = cat("right_children", np.int64)
    feat = cat("split_indices", np.int32)
    sconds = cat("split_conditions", np.float32)
    gain = cat("loss_changes", np.float32)
    cover = cat("sum_hessian", np.float32)
    del trees

    local_tree = np.repeat(np.arange(len(sizes)), sizes)
    is_leaf = left == -1
    gl = np.where(is_leaf, -1, left + offsets[local_tree])
    gr = np.where(is_leaf, -1, right + offsets[local_tree])

    # Breadth-first from every root at once; one numpy step per depth level.
    # Unreachable (pruned/deleted) nodes keep depth -1 and are dropped.
    depth = np.full(len(left), -1, dtype=np.int16)
    frontier, d = offsets.copy(), 0
    while frontier.size:
        depth[frontier] = d
        frontier = frontier[~is_leaf[frontier]]
        frontier = np.concatenate([gl[frontier], gr[frontier]])
        d += 1

    keep = depth >= 0
    nan = np.float32(np.nan)
    return pd.DataFrame(
        {
            "tree": (local_tree + tree_offset).astype(np.int32)[keep],
            "node": (np.arange(len(left)) - offsets[local_tree]).astype(np.int32)[keep],
            "depth": depth[keep],
            "is_leaf": is_leaf[keep],
            # child node ids within the same tree (trees_to_dataframe's Yes/No), -1 on leaves
            "left": left.astype(np.int32)[keep],
            "right": right.astype(np.int32)[keep],
            "feature": np.where(is_leaf, -1, feat)[keep],
            "split": np.where(is_leaf, nan, sconds)[keep],
            "gain": np.where(is_leaf, nan, gain)[keep],  # loss_chg, compare to gamma
            "cover": cover[keep],  # sum of hessian, compare to min_child_weight
            "leaf_value": np.where(is_leaf, sconds, nan)[keep],
        }
    )


def tree_node_stats(booster, rounds_per_chunk: int = 5) -> pd.DataFrame:
    """One row per reachable node of every tree in ``booster``.

    Columns: tree, node, depth, is_leaf, left, right, feature (index), split, gain,
    cover, leaf_value. Uses compact dtypes (~38 bytes per node). Lower
    ``rounds_per_chunk`` if a single chunk doesn't fit in memory.
    """
    parts, tree_offset = [], 0
    n_rounds = booster.num_boosted_rounds()
    for start in range(0, n_rounds, rounds_per_chunk):
        part = _chunk(
            booster[start : min(start + rounds_per_chunk, n_rounds)], tree_offset
        )
        tree_offset = int(part["tree"].max()) + 1
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _lambda_eta(booster):
    params = json.loads(booster.save_config())["learner"]["gradient_booster"][
        "tree_train_param"
    ]
    return float(params["lambda"]), float(params["eta"])


def add_gradient_statistics(
    stats: pd.DataFrame, lam: float, eta: float
) -> pd.DataFrame:
    """Add get_tree_statistics' G/H/Metric/Gain_Expected columns to tree_node_stats output.

    Vectorised equivalent of the per-tree loop in `get_tree_statistics`: leaf `G` is
    recovered from the leaf value (`-leaf_value * (H + lam) / eta`), then internal
    nodes are filled bottom-up one depth level at a time as `G[left] + G[right]`.

    Args:
        stats: output of `tree_node_stats` (needs tree, node, depth, is_leaf, left,
            right, cover, leaf_value).
        lam: the model's `lambda` (reg_lambda).
        eta: the model's learning rate.

    Returns:
        pandas DataFrame of `stats` plus `H`, `G`, `Metric` (= G^2 / (H + lam)),
        `Metric_Yes`, `Metric_No` (the children's Metric) and `Gain_Expected`
        (= Metric_Yes + Metric_No - Metric). The last three are NaN on leaves.
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
    return stats.assign(
        H=H,
        G=G,
        Metric=metric,
        Metric_Yes=metric_yes,
        Metric_No=metric_no,
        Gain_Expected=metric_yes + metric_no - metric,
    )


def get_tree_statistics_fast(model, rounds_per_chunk: int = 5) -> pd.DataFrame:
    """`get_tree_statistics` built on `tree_node_stats` instead of `trees_to_dataframe`.

    Same G/H/Metric/Gain_Expected values, but scales to large (e.g. depth-60) models.
    Columns follow tree_node_stats naming (tree, node, left, right, cover, ...)
    rather than trees_to_dataframe's (Tree, Node, Yes, No, Cover, ...).

    Args:
        model: a fitted XGBoost sklearn model, or a `xgb.Booster`.
        rounds_per_chunk: passed to `tree_node_stats`; lower it if memory is tight.

    Returns:
        pandas DataFrame, see `add_gradient_statistics`.
    """
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    lam, eta = _lambda_eta(booster)
    return add_gradient_statistics(tree_node_stats(booster, rounds_per_chunk), lam, eta)


# ---------------------------------------------------------------------------
# Reference implementation: polars on top of trees_to_dataframe (small models only)
# ---------------------------------------------------------------------------


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
