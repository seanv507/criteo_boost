import numpy as np
import pandas as pd
import polars as pl
import pytest
import xgboost as xgb

import tree_analysis as ta

# ---------------------------------------------------------------------------
# get_tree_statistics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=3, max_depth=2, tree_method="hist")
    clf.fit(X, y)
    return clf


def test_get_tree_statistics_schema_and_row_count(fitted_classifier):
    stats = ta.get_tree_statistics(fitted_classifier)
    raw = fitted_classifier.get_booster().trees_to_dataframe()

    assert stats.height == len(raw)
    for col in ("G", "H", "Metric", "Metric_Yes", "Metric_No", "Gain_Expected"):
        assert col in stats.columns


def test_get_tree_statistics_gradient_additivity(fitted_classifier):
    """An internal node's G must equal the sum of G at its two children."""
    stats = ta.get_tree_statistics(fitted_classifier)
    g_by_id = dict(zip(stats["ID"], stats["G"]))
    internal = stats.filter(pl.col("Feature") != "Leaf")

    assert internal.height > 0
    for row in internal.iter_rows(named=True):
        assert row["G"] == pytest.approx(g_by_id[row["Yes"]] + g_by_id[row["No"]])


def test_get_tree_statistics_gain_expected_consistent(fitted_classifier):
    stats = ta.get_tree_statistics(fitted_classifier)
    metric_by_id = dict(zip(stats["ID"], stats["Metric"]))
    internal = stats.filter(pl.col("Feature") != "Leaf")

    for row in internal.iter_rows(named=True):
        expected = metric_by_id[row["Yes"]] + metric_by_id[row["No"]] - row["Metric"]
        assert row["Gain_Expected"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# get_tree_statistics_fast / add_gradient_statistics
# ---------------------------------------------------------------------------

STAT_COLS = ["H", "G", "Metric", "Metric_Yes", "Metric_No", "Gain_Expected"]


@pytest.fixture(scope="module")
def deep_regularised_classifier():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 6))
    y = (X[:, 0] * X[:, 1] + np.sin(X[:, 2]) > 0).astype(int)
    clf = xgb.XGBClassifier(
        n_estimators=7,
        max_depth=8,
        learning_rate=0.3,
        reg_lambda=3.5,
        min_child_weight=0.1,
        tree_method="hist",
    )
    clf.fit(X, y)
    return clf


def _compare_with_slow(model, rounds_per_chunk):
    # the polars reference's nulls become NaN in pandas, matching the fast path
    slow = (
        ta.get_tree_statistics(model)
        .select(pl.col("Tree").alias("tree"), pl.col("Node").alias("node"), *STAT_COLS)
        .to_pandas()
    )
    fast = ta.get_tree_statistics_fast(model, rounds_per_chunk=rounds_per_chunk)
    assert isinstance(fast, pd.DataFrame)
    assert len(fast) == len(slow)
    joined = fast[["tree", "node", *STAT_COLS]].merge(
        slow, on=["tree", "node"], suffixes=("", "_slow"), validate="1:1"
    )
    assert len(joined) == len(slow)
    for col in STAT_COLS:
        # NaNs (leaves' Metric_Yes/No/Gain_Expected) must line up too
        assert (joined[col].isna() == joined[f"{col}_slow"].isna()).all()
        both = joined.dropna(subset=[col, f"{col}_slow"])
        # fast reads float32 node arrays, slow goes via trees_to_dataframe's floats
        np.testing.assert_allclose(
            both[col].to_numpy(), both[f"{col}_slow"].to_numpy(), rtol=1e-4, atol=1e-3
        )


@pytest.mark.parametrize("rounds_per_chunk", [1, 5])
def test_get_tree_statistics_fast_matches_slow(fitted_classifier, rounds_per_chunk):
    _compare_with_slow(fitted_classifier, rounds_per_chunk)


@pytest.mark.parametrize("rounds_per_chunk", [1, 3, 100])
def test_get_tree_statistics_fast_matches_slow_deep_regularised(
    deep_regularised_classifier, rounds_per_chunk
):
    _compare_with_slow(deep_regularised_classifier, rounds_per_chunk)


def test_get_tree_statistics_fast_accepts_booster(fitted_classifier):
    from_model = ta.get_tree_statistics_fast(fitted_classifier)
    from_booster = ta.get_tree_statistics_fast(fitted_classifier.get_booster())
    assert from_model.equals(from_booster)


def test_add_gradient_statistics_hand_built_tree():
    # one tree: root 0 -> (1, 2); node 1 -> leaves (3, 4); node 2 is a leaf
    stats = pd.DataFrame(
        {
            "tree": [0, 0, 0, 0, 0],
            "node": [0, 1, 2, 3, 4],
            "depth": [0, 1, 1, 2, 2],
            "is_leaf": [False, False, True, True, True],
            "left": [1, 3, -1, -1, -1],
            "right": [2, 4, -1, -1, -1],
            "cover": [10.0, 6.0, 4.0, 2.0, 4.0],
            "leaf_value": [np.nan, np.nan, 0.5, -0.25, 0.1],
        }
    )
    lam, eta = 1.0, 0.5
    out = ta.add_gradient_statistics(stats, lam, eta)

    g_leaf = {
        n: -v * (h + lam) / eta for n, v, h in [(2, 0.5, 4), (3, -0.25, 2), (4, 0.1, 4)]
    }
    g1 = g_leaf[3] + g_leaf[4]
    g0 = g1 + g_leaf[2]
    assert out["G"].to_list() == pytest.approx(
        [g0, g1, g_leaf[2], g_leaf[3], g_leaf[4]]
    )
    metric = [g**2 / (h + lam) for g, h in zip(out["G"], stats["cover"])]
    assert out["Metric"].to_list() == pytest.approx(metric)
    assert out["Gain_Expected"][0] == pytest.approx(metric[1] + metric[2] - metric[0])
    assert out["Gain_Expected"][1] == pytest.approx(metric[3] + metric[4] - metric[1])
    assert out["Gain_Expected"][2:].isna().all()
    assert out["Metric_Yes"].tolist()[:2] == pytest.approx([metric[1], metric[3]])
    assert out["Metric_No"].tolist()[:2] == pytest.approx([metric[2], metric[4]])
    # input columns pass through unchanged, the input frame isn't modified
    pd.testing.assert_frame_equal(out[stats.columns], stats)
    assert "G" not in stats.columns


# ---------------------------------------------------------------------------
# tree_node_stats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rounds_per_chunk", [1, 2, 100])
def test_tree_node_stats_matches_trees_to_dataframe(
    deep_regularised_classifier, rounds_per_chunk
):
    booster = deep_regularised_classifier.get_booster()
    stats = ta.tree_node_stats(booster, rounds_per_chunk=rounds_per_chunk)
    ref = booster.trees_to_dataframe()

    assert isinstance(stats, pd.DataFrame)
    assert list(stats.columns) == [
        "tree",
        "node",
        "depth",
        "is_leaf",
        "left",
        "right",
        "missing",
        "feature",
        "split",
        "gain",
        "cover",
        "leaf_value",
    ]
    stats = stats.sort_values(["tree", "node"]).reset_index(drop=True)
    ref = ref.sort_values(["Tree", "Node"]).reset_index(drop=True)
    assert len(stats) == len(ref)
    assert (stats["tree"].to_numpy() == ref["Tree"].to_numpy()).all()
    assert (stats["node"].to_numpy() == ref["Node"].to_numpy()).all()

    leaf = (ref["Feature"] == "Leaf").to_numpy()
    assert (stats["is_leaf"].to_numpy() == leaf).all()
    np.testing.assert_allclose(stats["cover"], ref["Cover"], rtol=1e-6)
    np.testing.assert_allclose(stats["gain"][~leaf], ref["Gain"][~leaf], rtol=1e-6)
    np.testing.assert_allclose(stats["leaf_value"][leaf], ref["Gain"][leaf], rtol=1e-6)
    np.testing.assert_allclose(stats["split"][~leaf], ref["Split"][~leaf], rtol=1e-6)
    assert stats.loc[leaf, ["gain", "split"]].isna().all().all()
    assert stats.loc[~leaf, "leaf_value"].isna().all()

    # children: trees_to_dataframe's Yes/No are "tree-node" ids
    child = lambda col: ref[col][~leaf].str.split("-").str[1].astype(int).to_numpy()
    assert (stats["left"][~leaf].to_numpy() == child("Yes")).all()
    assert (stats["right"][~leaf].to_numpy() == child("No")).all()
    assert (stats["missing"][~leaf].to_numpy() == child("Missing")).all()
    assert (stats.loc[leaf, ["left", "right", "missing"]] == -1).all().all()


def test_tree_node_stats_missing_follows_default_direction():
    # NaNs in the training data make XGBoost learn a default direction per split
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X[rng.random(X.shape) < 0.3] = np.nan
    X[np.isnan(X[:, 0]) & (y == 1), 1] = np.nan  # make missingness informative
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=4, tree_method="hist")
    clf.fit(X, y)
    booster = clf.get_booster()

    stats = ta.tree_node_stats(booster, rounds_per_chunk=2)
    stats = stats.sort_values(["tree", "node"]).reset_index(drop=True)
    ref = booster.trees_to_dataframe().sort_values(["Tree", "Node"])
    internal = (ref["Feature"] != "Leaf").to_numpy()
    expected = ref["Missing"][internal].str.split("-").str[1].astype(int).to_numpy()

    split_rows = stats[~stats["is_leaf"]]
    assert (split_rows["missing"].to_numpy() == expected).all()
    goes_left = split_rows["missing"] == split_rows["left"]
    goes_right = split_rows["missing"] == split_rows["right"]
    assert (goes_left | goes_right).all()
    assert goes_left.any() and goes_right.any()  # both directions are exercised


def test_tree_node_stats_depth_and_tree_numbering(deep_regularised_classifier):
    stats = ta.tree_node_stats(deep_regularised_classifier.get_booster(), 3)
    n_trees = deep_regularised_classifier.get_booster().num_boosted_rounds()
    assert sorted(stats["tree"].unique()) == list(range(n_trees))
    assert (stats.loc[stats["node"] == 0, "depth"] == 0).all()
    depth = stats.set_index(["tree", "node"])["depth"]
    internal = stats[~stats["is_leaf"]]
    for col in ("left", "right"):
        child_depth = depth.loc[list(zip(internal["tree"], internal[col]))].to_numpy()
        assert (child_depth == internal["depth"].to_numpy() + 1).all()


# ---------------------------------------------------------------------------
# add_level
# ---------------------------------------------------------------------------


def test_add_level_assigns_depth_from_root(fitted_classifier):
    df = pl.from_pandas(fitted_classifier.get_booster().trees_to_dataframe())
    out = ta.add_level(df)

    assert out.height == df.height
    level_by_id = dict(zip(out["ID"], out["Level"]))
    assert set(out.filter(pl.col("Node") == 0)["Level"]) == {0}
    for row in out.filter(pl.col("Feature") != "Leaf").iter_rows(named=True):
        assert level_by_id[row["Yes"]] == row["Level"] + 1
        assert level_by_id[row["No"]] == row["Level"] + 1
