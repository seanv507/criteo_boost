"""Fast per-node statistics for an XGBoost tree model.

Vectorised with numpy, and processes the model a few boosting rounds at a time
(``booster[a:b]``), so peak memory is bounded by one chunk of trees rather than
by the whole model. The result uses compact dtypes (~30 bytes per node).
"""

import json

import numpy as np
import pandas as pd


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
            "feature": np.where(is_leaf, -1, feat)[keep],
            "split": np.where(is_leaf, nan, sconds)[keep],
            "gain": np.where(is_leaf, nan, gain)[keep],  # loss_chg, compare to gamma
            "cover": cover[keep],  # sum of hessian, compare to min_child_weight
            "leaf_value": np.where(is_leaf, sconds, nan)[keep],
        }
    )


def tree_node_stats(booster, rounds_per_chunk: int = 5) -> pd.DataFrame:
    """One row per reachable node of every tree in ``booster``.

    Columns: tree, node, depth, is_leaf, feature (index), split, gain, cover,
    leaf_value. Lower ``rounds_per_chunk`` if a single chunk doesn't fit in memory.
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
