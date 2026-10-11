import json

import polars as pl
import xgboost as xgb


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

    lam = float(
        json.loads(booster.save_config())["learner"]["gradient_booster"][
            "tree_train_param"
        ]["lambda"]
    )
    eta = float(
        json.loads(booster.save_config())["learner"]["gradient_booster"][
            "tree_train_param"
        ]["eta"]
    )
    df = df.with_columns(H=pl.col("Cover"))  # hessian sum, valid for every row
    is_leaf = pl.col("Feature") == "Leaf"
    df = df.with_columns(
        G=pl.when(is_leaf).then(-pl.col("Gain") * (pl.col("H") + lam) / eta)
    )
    tdfs = {}
    for tree_id, tdf in df.sort("Node", descending=True).group_by("Tree"):
        print(tree_id)
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
    df_roots = df.filter(pl.col("Node") == 0)
    df_roots["Level"] = 0
    id_next = df.select(["Yes", "No"]).unpivot(on=["Yes", "No"])
    ### WIP
    return id_next
