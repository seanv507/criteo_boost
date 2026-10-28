import numpy as np
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
