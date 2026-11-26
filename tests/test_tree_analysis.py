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
    slow = ta.get_tree_statistics(model).select(
        pl.col("Tree").alias("tree"), pl.col("Node").alias("node"), *STAT_COLS
    )
    fast = ta.get_tree_statistics_fast(model, rounds_per_chunk=rounds_per_chunk)
    assert fast.height == slow.height
    joined = fast.select("tree", "node", *STAT_COLS).join(
        slow, on=["tree", "node"], suffix="_slow", validate="1:1"
    )
    assert joined.height == slow.height
    for col in STAT_COLS:
        # nulls (leaves' Metric_Yes/No/Gain_Expected) must line up too
        assert (
            joined[col].is_null().to_list() == joined[f"{col}_slow"].is_null().to_list()
        )
        both = joined.drop_nulls([col, f"{col}_slow"])
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
    assert out["Gain_Expected"][2:].is_null().all()


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
