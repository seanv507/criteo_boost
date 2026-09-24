import math

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

import criteo_functions as cf


# ---------------------------------------------------------------------------
# create_ctr
# ---------------------------------------------------------------------------

def test_create_ctr_matches_formula():
    df = pl.LazyFrame({
        "g": ["X"] * 4 + ["Y"] * 4,
        "Label": [1, 1, 0, 0, 1, 0, 0, 0],
    })
    out = cf.create_ctr(df, "g").collect().sort("g")

    ctr_global = 3 / 8
    for row in out.iter_rows(named=True):
        expected_margin = 1.96 * math.sqrt(ctr_global * (1 - ctr_global) / row["count"])
        expected_diff = row["ctr"] - ctr_global
        assert row["diff"] == pytest.approx(expected_diff)
        assert row["margin"] == pytest.approx(expected_margin)
        assert row["margin_n"] == pytest.approx(-expected_margin)
        assert row["delta"] == pytest.approx(abs(expected_diff) / expected_margin)

    assert set(out["g"]) == {"X", "Y"}
    x_row = out.filter(pl.col("g") == "X").row(0, named=True)
    assert x_row["ctr"] == pytest.approx(0.5)


def test_create_ctr_sorted_by_count_descending():
    df = pl.LazyFrame({
        "g": ["small"] * 2 + ["big"] * 5,
        "Label": [1, 0, 1, 1, 0, 0, 1],
    })
    out = cf.create_ctr(df, "g").collect()
    assert out["g"].to_list() == ["big", "small"]


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------

def test_fill_missing_replaces_nulls_only_in_given_cols():
    df = pl.LazyFrame({
        "a": ["x", None, "z"],
        "b": ["p", None, "r"],
    })
    out = cf.fill_missing(df, ["a"]).collect()
    assert out["a"].to_list() == ["x", "missing", "z"]
    assert out["b"].to_list() == ["p", None, "r"]


def test_fill_missing_custom_value():
    df = pl.LazyFrame({"a": [None, "y"]})
    out = cf.fill_missing(df, ["a"], missing_val="unknown").collect()
    assert out["a"].to_list() == ["unknown", "y"]


# ---------------------------------------------------------------------------
# transform_cat
# ---------------------------------------------------------------------------

def test_transform_cat_smoothing_and_cutoff():
    df = pl.LazyFrame({
        "cat": ["a", "a", "a", "a", "a", "b", "b", "c"],
        "Label": [1, 1, 0, 0, 0, 1, 0, 1],
    })
    df_ctr, rest, global_mean = cf.transform_cat(df, "cat", "Label", cutoff=2, pseudo_count=2)
    df_ctr = df_ctr.collect()
    rest = rest.collect()

    assert global_mean == pytest.approx(0.5)

    # 'c' has count 1 < cutoff=2, so it is pooled into `rest`
    assert set(df_ctr["cat"]) == {"a", "b"}
    rest_row = rest.row(0, named=True)
    assert rest_row["count"] == 1
    assert rest_row["clicks"] == 1
    assert rest_row["ctr"] == pytest.approx(1.0)
    assert rest_row["ctr_smooth"] == pytest.approx((1 + 0.5 * 2) / (1 + 2))

    a_row = df_ctr.filter(pl.col("cat") == "a").row(0, named=True)
    expected_smooth_a = (2 + 0.5 * 2) / (5 + 2)
    assert a_row["ctr_smooth_diff"] == pytest.approx(expected_smooth_a - 0.5)

    b_row = df_ctr.filter(pl.col("cat") == "b").row(0, named=True)
    expected_smooth_b = (1 + 0.5 * 2) / (2 + 2)
    assert b_row["ctr_smooth_diff"] == pytest.approx(expected_smooth_b - 0.5)

    # sorted descending by ctr_smooth_diff
    assert df_ctr["ctr_smooth_diff"].to_list() == sorted(
        df_ctr["ctr_smooth_diff"].to_list(), reverse=True
    )


def test_transform_cat_all_rare_categories_pooled():
    df = pl.LazyFrame({"cat": ["a", "b", "c"], "Label": [1, 0, 1]})
    df_ctr, rest, global_mean = cf.transform_cat(df, "cat", "Label", cutoff=10, pseudo_count=1)
    assert df_ctr.collect().is_empty()
    rest_row = rest.collect().row(0, named=True)
    assert rest_row["count"] == 3
    assert rest_row["clicks"] == 2
    assert global_mean == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# calc_cat_encode
# ---------------------------------------------------------------------------

def test_calc_cat_encode_returns_one_table_per_column():
    df = pl.LazyFrame({
        "C1": ["a", "a", "b", "b", "c"],
        "C2": ["x", "x", "x", "y", "y"],
        "Label": [1, 0, 1, 0, 1],
    })
    config = {"cat_cutoff": 1, "cat_pseudo_count": 1}
    cat_encode, global_mean = cf.calc_cat_encode(df, config, ["C1", "C2"])

    assert set(cat_encode.keys()) == {"C1", "C2"}
    assert global_mean == pytest.approx(3 / 5)
    # cutoff=1 keeps every category (all counts >= 1), plus one pooled "rest" row
    # with a null category value, from the diagonal concat.
    assert cat_encode["C1"].height == 3 + 1
    assert cat_encode["C2"].height == 2 + 1
    assert "ctr_smooth_diff" in cat_encode["C1"].columns


# ---------------------------------------------------------------------------
# encode_cats
# ---------------------------------------------------------------------------

def test_encode_cats_joins_and_fills_unmatched_with_zero():
    cat_encode = {
        "C1": pl.DataFrame({"C1": ["a", "b"], "ctr_smooth_diff": [0.1, -0.2]}),
    }
    df = pl.LazyFrame({"C1": ["a", "b", "c"], "Label": [1, 0, 1]})
    out = cf.encode_cats(df, cat_encode, ["C1"]).collect()

    assert "C1" not in out.columns
    assert "C1_encode" in out.columns
    assert out["C1_encode"].to_list() == pytest.approx([0.1, -0.2, 0.0])


def test_encode_cats_multiple_columns():
    cat_encode = {
        "C1": pl.DataFrame({"C1": ["a"], "ctr_smooth_diff": [0.5]}),
        "C2": pl.DataFrame({"C2": ["x"], "ctr_smooth_diff": [-0.5]}),
    }
    df = pl.LazyFrame({"C1": ["a"], "C2": ["x"], "Label": [1]})
    out = cf.encode_cats(df, cat_encode, ["C1", "C2"]).collect()
    assert out["C1_encode"].to_list() == [0.5]
    assert out["C2_encode"].to_list() == [-0.5]


# ---------------------------------------------------------------------------
# get_data_XY
# ---------------------------------------------------------------------------

def test_get_data_xy_selects_expected_columns():
    df = pl.LazyFrame({
        "Row": [0, 1],
        "Label": [1, 0],
        "I1": [1.0, 2.0],
        "I2": [3.0, 4.0],
        "C1_encode": [0.1, 0.2],
        "ar_flag": [1, 0],
        "other": ["x", "y"],
    })
    data_X, data_y = cf.get_data_XY(df)

    assert set(data_X.columns) == {"I1", "I2", "C1_encode", "ar_flag"}
    assert data_y.to_list() == [1, 0]
    assert data_X.height == 2


def test_get_data_xy_works_without_row_column():
    df = pl.LazyFrame({"Label": [1], "I1": [5.0]})
    data_X, data_y = cf.get_data_XY(df)
    assert data_X.columns == ["I1"]
    assert data_y.to_list() == [1]


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
    stats = cf.get_tree_statistics(fitted_classifier)
    raw = fitted_classifier.get_booster().trees_to_dataframe()

    assert stats.height == len(raw)
    for col in ("G", "H", "Metric", "Metric_Yes", "Metric_No", "Gain_Expected"):
        assert col in stats.columns


def test_get_tree_statistics_gradient_additivity(fitted_classifier):
    """An internal node's G must equal the sum of G at its two children."""
    stats = cf.get_tree_statistics(fitted_classifier)
    g_by_id = dict(zip(stats["ID"], stats["G"]))
    internal = stats.filter(pl.col("Feature") != "Leaf")

    assert internal.height > 0
    for row in internal.iter_rows(named=True):
        assert row["G"] == pytest.approx(g_by_id[row["Yes"]] + g_by_id[row["No"]])


def test_get_tree_statistics_gain_expected_consistent(fitted_classifier):
    stats = cf.get_tree_statistics(fitted_classifier)
    metric_by_id = dict(zip(stats["ID"], stats["Metric"]))
    internal = stats.filter(pl.col("Feature") != "Leaf")

    for row in internal.iter_rows(named=True):
        expected = metric_by_id[row["Yes"]] + metric_by_id[row["No"]] - row["Metric"]
        assert row["Gain_Expected"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# create_expression
# ---------------------------------------------------------------------------

def test_create_expression_filters_matching_rows():
    df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    expr = cf.create_expression([("a", 1), ("b", "x")])
    out = df.filter(expr)
    assert out.height == 1
    assert out.row(0) == (1, "x")


def test_create_expression_single_pair():
    df = pl.DataFrame({"a": [1, 2, 3]})
    expr = cf.create_expression([("a", 2)])
    out = df.filter(expr)
    assert out["a"].to_list() == [2]


# ---------------------------------------------------------------------------
# parse_frequent
# ---------------------------------------------------------------------------

def test_parse_frequent_parses_items_and_support(tmp_path):
    freq_file = tmp_path / "freq.txt"
    freq_file.write_text("C1:v1 C2:v2 (12.34)\nC3:v3 (5.00)\n")

    df = cf.parse_frequent(str(freq_file)).collect()

    assert df.height == 2
    assert df["support"].to_list() == pytest.approx([0.1234, 0.05])
    assert df["text"].to_list() == [[["C1", "v1"], ["C2", "v2"]], [["C3", "v3"]]]


def test_parse_frequent_rules_filter_correctly(tmp_path):
    freq_file = tmp_path / "freq.txt"
    freq_file.write_text("C1:v1 C2:v2 (12.34)\n")

    rule = cf.parse_frequent(str(freq_file)).collect()["rules"][0]
    matching = pl.DataFrame({"C1": ["v1", "other"], "C2": ["v2", "v2"]})
    assert matching.filter(rule).height == 1


# ---------------------------------------------------------------------------
# parse_association
# ---------------------------------------------------------------------------

def test_parse_association_parses_items_support_confidence(tmp_path):
    assoc_file = tmp_path / "assoc.txt"
    assoc_file.write_text("Label:1 <- C1:v1 C2:v2 (12.34, 56.78)\n")

    df = cf.parse_association(str(assoc_file)).collect()

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["items"] == [["C1", "v1"], ["C2", "v2"]]
    assert row["support"] == pytest.approx(0.1234)
    assert row["confidence"] == pytest.approx(0.5678)


def test_parse_association_rules_filter_correctly(tmp_path):
    assoc_file = tmp_path / "assoc.txt"
    assoc_file.write_text("Label:1 <- C1:v1 (10.00, 20.00)\n")

    rule = cf.parse_association(str(assoc_file)).collect()["rules"][0]
    matching = pl.DataFrame({"C1": ["v1", "other"]})
    assert matching.filter(rule).height == 1


# ---------------------------------------------------------------------------
# build_categorical_enums
# ---------------------------------------------------------------------------

def test_build_categorical_enums_unions_sorted_distinct_values(tmp_path):
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    pl.DataFrame({"C1": ["x", "y", None]}).write_parquet(path_a)
    pl.DataFrame({"C1": ["y", "z"]}).write_parquet(path_b)

    enums = cf.build_categorical_enums([path_a, path_b], cat_cols=["C1"])

    assert set(enums.keys()) == {"C1"}
    assert isinstance(enums["C1"], pl.Enum)
    assert enums["C1"].categories.to_list() == ["x", "y", "z"]


def test_build_categorical_enums_multiple_columns(tmp_path):
    path_a = tmp_path / "a.parquet"
    pl.DataFrame({"C1": ["b", "a"], "C2": ["q", "p"]}).write_parquet(path_a)

    enums = cf.build_categorical_enums([path_a], cat_cols=["C1", "C2"])

    assert enums["C1"].categories.to_list() == ["a", "b"]
    assert enums["C2"].categories.to_list() == ["p", "q"]


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_subsamples_train_only(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "Criteo"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"x": list(range(6))}).write_parquet(data_dir / "train.parquet")
    pl.DataFrame({"x": list(range(3))}).write_parquet(data_dir / "valid.parquet")
    pl.DataFrame({"x": list(range(2))}).write_parquet(data_dir / "test.parquet")

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    train, valid, test = cf.load_data(sample_every=2)

    train_df = train.collect()
    assert train_df.height == 3
    assert train_df["x"].to_list() == [0, 2, 4]
    assert valid.collect().height == 3
    assert test.collect().height == 2


def test_load_data_accepts_explicit_data_dir(tmp_path):
    data_dir = tmp_path / "somewhere" / "else"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"x": list(range(4))}).write_parquet(data_dir / "train.parquet")
    pl.DataFrame({"x": list(range(3))}).write_parquet(data_dir / "valid.parquet")
    pl.DataFrame({"x": list(range(2))}).write_parquet(data_dir / "test.parquet")

    train, valid, test = cf.load_data(sample_every=2, data_dir=data_dir)

    assert train.collect()["x"].to_list() == [0, 2]
    assert valid.collect().height == 3
    assert test.collect().height == 2


# ---------------------------------------------------------------------------
# get_data
# ---------------------------------------------------------------------------

def _synthetic_split(n_rows, offset=0):
    n_ints, n_cats = 13, 26
    data = {
        "Row": list(range(offset, offset + n_rows)),
        "Label": [(i + offset) % 2 for i in range(n_rows)],
    }
    for i in range(1, n_ints + 1):
        data[f"I{i}"] = [float(i + row) for row in range(n_rows)]
    for i in range(1, n_cats + 1):
        data[f"C{i}"] = [f"v{(row + i) % 3}" for row in range(n_rows)]
    # introduce a null to exercise fill_missing
    data["C1"][0] = None
    return pl.LazyFrame(data)


def test_get_data_produces_expected_shapes(monkeypatch):
    train, valid, test = _synthetic_split(6), _synthetic_split(3, offset=100), _synthetic_split(2, offset=200)
    monkeypatch.setattr(cf, "load_data", lambda sample_every, data_dir=None: (train, valid, test))

    config = {"sample_every": 1, "cat_cutoff": 1, "cat_pseudo_count": 1}
    train_X, train_y, valid_X, valid_y, test_X, test_y = cf.get_data(config)

    expected_cols = {f"I{i}" for i in range(1, 14)} | {f"C{i}_encode" for i in range(1, 27)}
    assert set(train_X.columns) == expected_cols
    assert train_X.height == 6
    assert len(train_y) == 6
    assert valid_X.height == 3
    assert len(valid_y) == 3
    assert test_X.height == 2
    assert len(test_y) == 2
