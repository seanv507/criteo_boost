import math
from pathlib import Path

import polars as pl
import pytest

import criteo_functions as cf

# ---------------------------------------------------------------------------
# create_ctr
# ---------------------------------------------------------------------------


def test_create_ctr_matches_formula():
    df = pl.LazyFrame(
        {
            "g": ["X"] * 4 + ["Y"] * 4,
            "Label": [1, 1, 0, 0, 1, 0, 0, 0],
        }
    )
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
    df = pl.LazyFrame(
        {
            "g": ["small"] * 2 + ["big"] * 5,
            "Label": [1, 0, 1, 1, 0, 0, 1],
        }
    )
    out = cf.create_ctr(df, "g").collect()
    assert out["g"].to_list() == ["big", "small"]


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------


def test_fill_missing_replaces_nulls_only_in_given_cols():
    df = pl.LazyFrame(
        {
            "a": ["x", None, "z"],
            "b": ["p", None, "r"],
        }
    )
    out = cf.fill_missing(df, ["a"]).collect()
    assert out["a"].to_list() == ["x", "missing", "z"]
    assert out["b"].to_list() == ["p", None, "r"]


def test_fill_missing_custom_value():
    df = pl.LazyFrame({"a": [None, "y"]})
    out = cf.fill_missing(df, ["a"], missing_val="unknown").collect()
    assert out["a"].to_list() == ["unknown", "y"]


# ---------------------------------------------------------------------------
# calc_cat_smooth_ctr
# ---------------------------------------------------------------------------


def test_calc_cat_smooth_ctr_smoothing_and_cutoff():
    df = pl.LazyFrame(
        {
            "cat": ["a", "a", "a", "a", "a", "b", "b", "c"],
            "Label": [1, 1, 0, 0, 0, 1, 0, 1],
        }
    )
    df_ctr, rest, global_mean = cf.calc_cat_smooth_ctr(
        df, "cat", "Label", cutoff=2, pseudo_count=2
    )
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


def test_calc_cat_smooth_ctr_all_rare_categories_pooled():
    df = pl.LazyFrame({"cat": ["a", "b", "c"], "Label": [1, 0, 1]})
    df_ctr, rest, global_mean = cf.calc_cat_smooth_ctr(
        df, "cat", "Label", cutoff=10, pseudo_count=1
    )
    assert df_ctr.collect().is_empty()
    rest_row = rest.collect().row(0, named=True)
    assert rest_row["count"] == 3
    assert rest_row["clicks"] == 2
    assert global_mean == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# calc_cat_encode
# ---------------------------------------------------------------------------


def test_calc_cat_encode_returns_one_table_per_column():
    df = pl.LazyFrame(
        {
            "C1": ["a", "a", "b", "b", "c"],
            "C2": ["x", "x", "x", "y", "y"],
            "Label": [1, 0, 1, 0, 1],
        }
    )
    cat_encode, global_mean = cf.calc_cat_encode(
        df, ["C1", "C2"], cutoff=1, pseudo_count=1
    )

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
    out = cf.encode_cats(df, cat_encode, ["C1"]).collect().sort("C1")

    # the encoded value replaces the category under the original column name
    assert set(out.columns) == {"C1", "Label"}
    assert out["C1"].dtype == pl.Float64
    assert out["C1"].to_list() == pytest.approx([-0.2, 0.0, 0.1])
    # row order is not preserved by the join, so check the label alignment too
    assert out["Label"].to_list() == [0, 1, 1]


def test_encode_cats_multiple_columns():
    cat_encode = {
        "C1": pl.DataFrame({"C1": ["a"], "ctr_smooth_diff": [0.5]}),
        "C2": pl.DataFrame({"C2": ["x"], "ctr_smooth_diff": [-0.5]}),
    }
    df = pl.LazyFrame({"C1": ["a"], "C2": ["x"], "Label": [1]})
    out = cf.encode_cats(df, cat_encode, ["C1", "C2"]).collect()
    assert out["C1"].to_list() == [0.5]
    assert out["C2"].to_list() == [-0.5]


# ---------------------------------------------------------------------------
# merge_rare / create_enums / cast_enums
# ---------------------------------------------------------------------------


def _cat_encode_with_rest():
    # shape produced by calc_cat_encode: kept categories plus a null-category "rest" row
    return {
        "C1": pl.DataFrame(
            {
                "C1": ["a", "b", None],
                "count": [5, 9, 2],
                "ctr_smooth_diff": [0.1, 0.2, None],
            }
        ),
    }


def test_merge_rare_maps_unknown_categories_to_rare():
    df = pl.LazyFrame({"C1": ["a", "b", "zzz"], "Label": [1, 0, 1]})
    out = cf.merge_rare(df, _cat_encode_with_rest()).collect()
    assert out["C1"].to_list() == ["a", "b", "rare"]
    assert out["Label"].to_list() == [1, 0, 1]


def test_create_enums_orders_by_count_and_appends_rare():
    enums = cf.create_enums(_cat_encode_with_rest())
    assert set(enums.keys()) == {"C1"}
    assert isinstance(enums["C1"], pl.Enum)
    # null "rest" row dropped, most frequent first, "rare" last
    assert enums["C1"].categories.to_list() == ["b", "a", "rare"]


def test_cast_enums_casts_columns():
    enums = cf.create_enums(_cat_encode_with_rest())
    df = pl.LazyFrame({"C1": ["a", "rare", "b"], "I1": [1.0, 2.0, 3.0]})
    out = cf.cast_enums(df, enums).collect()
    assert out["C1"].dtype == enums["C1"]
    assert out["C1"].to_list() == ["a", "rare", "b"]
    assert out["I1"].dtype == pl.Float64


# ---------------------------------------------------------------------------
# get_data_XY
# ---------------------------------------------------------------------------


def test_get_data_xy_drops_row_and_label():
    df = pl.LazyFrame(
        {
            "Row": [0, 1],
            "Label": [1, 0],
            "I1": [1.0, 2.0],
            "I2": [3.0, 4.0],
            "C1": [0.1, 0.2],
            "other": ["x", "y"],
        }
    )
    data_X, data_y = cf.get_data_XY(df)

    assert set(data_X.columns) == {"I1", "I2", "C1", "other"}
    assert data_y.to_list() == [1, 0]
    assert data_X.height == 2


def test_get_data_xy_works_without_row_column():
    df = pl.LazyFrame({"Label": [1], "I1": [5.0]})
    data_X, data_y = cf.get_data_XY(df)
    assert data_X.columns == ["I1"]
    assert data_y.to_list() == [1]


# ---------------------------------------------------------------------------
# select_inputs
# ---------------------------------------------------------------------------


def _model_frames():
    data = {col: [1.0] for col in cf.INTS} | {col: [0.5] for col in cf.CAT_COLS}
    return [pl.DataFrame(data), pl.DataFrame(data)]


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ("ints", cf.INTS),
        ("cats", cf.CAT_COLS),
        ("ints,cats", cf.INTS + cf.CAT_COLS),
        (" ints , C1,C5 ", cf.INTS + ["C1", "C5"]),
        ("C5,I1", ["C5", "I1"]),
        ("C1,cats", cf.CAT_COLS),  # duplicates dropped, first occurrence wins
    ],
)
def test_select_inputs_groups_and_names(inputs, expected):
    out = cf.select_inputs(_model_frames(), inputs)
    assert len(out) == 2
    for df in out:
        assert df.columns == expected


@pytest.mark.parametrize("inputs", ["all", None, "ints,all"])
def test_select_inputs_all_keeps_every_column(inputs):
    frames = _model_frames()
    out = cf.select_inputs(frames, inputs)
    assert [df.columns for df in out] == [df.columns for df in frames]


def test_select_inputs_works_on_tuple_of_lazyframes():
    frames = tuple(df.lazy() for df in _model_frames())
    out = cf.select_inputs(frames, "I2")
    assert [df.collect_schema().names() for df in out] == [["I2"], ["I2"]]


def test_select_inputs_empty_spec_raises():
    with pytest.raises(ValueError, match="selects no columns"):
        cf.select_inputs(_model_frames(), " , ")


def test_select_inputs_unknown_column_raises():
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        cf.select_inputs(_model_frames(), "ints,nope")


# ---------------------------------------------------------------------------
# notify
# ---------------------------------------------------------------------------


def test_notify_noop_without_topic(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.setattr(cf.requests, "post", lambda *a, **k: pytest.fail("posted"))
    cf.notify("done")


def test_notify_posts_to_topic(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    calls = []
    monkeypatch.setattr(cf.requests, "post", lambda url, **k: calls.append((url, k)))
    cf.notify("done", title="T")
    assert calls == [
        (
            "https://ntfy.sh/my-topic",
            {"data": b"done", "headers": {"Title": "T"}, "timeout": 10},
        )
    ]


def test_notify_swallows_request_errors(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")

    def boom(*a, **k):
        raise cf.requests.ConnectionError("offline")

    monkeypatch.setattr(cf.requests, "post", boom)
    cf.notify("done")  # must not raise


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


def test_load_data_subsamples_train_only(tmp_path):
    data_dir = tmp_path / "somewhere" / "else"
    data_dir.mkdir(parents=True)
    pl.DataFrame({"x": list(range(6))}).write_parquet(data_dir / "train.parquet")
    pl.DataFrame({"x": list(range(3))}).write_parquet(data_dir / "valid.parquet")
    pl.DataFrame({"x": list(range(2))}).write_parquet(data_dir / "test.parquet")

    train, valid, test = cf.load_data(sample_every=2, data_dir=data_dir)

    train_df = train.collect()
    assert train_df["x"].to_list() == [0, 2, 4]
    assert train_df["Row"].to_list() == [0, 2, 4]
    assert valid.collect().height == 3
    assert test.collect().height == 2


# ---------------------------------------------------------------------------
# get_data_dir
# ---------------------------------------------------------------------------


def test_get_data_dir_prefers_home_projects_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    home_dir = tmp_path / "projects" / "data" / "Criteo"
    home_dir.mkdir(parents=True)
    assert cf.get_data_dir() == home_dir


def test_get_data_dir_falls_back_to_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/projects/data/Criteo here
    assert cf.get_data_dir() == Path("/workspace/data/Criteo")


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


@pytest.fixture
def patched_load_data(monkeypatch):
    splits = (
        _synthetic_split(6),
        _synthetic_split(3, offset=100),
        _synthetic_split(2, offset=200),
    )
    calls = []

    def fake_load_data(sample_every, data_dir):
        calls.append((sample_every, data_dir))
        return splits

    monkeypatch.setattr(cf, "load_data", fake_load_data)
    return calls


EXPECTED_COLS = set(cf.INTS) | set(cf.CAT_COLS)


def _check_shapes(datasets):
    (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = datasets
    for X, y, n in ((train_X, train_y, 6), (valid_X, valid_y, 3), (test_X, test_y, 2)):
        assert set(X.columns) == EXPECTED_COLS
        assert X.height == n
        assert len(y) == n


def test_get_data_target_encoding(patched_load_data):
    config = {
        "sample_every": 3,
        "cat_cutoff": 1,
        "cat_pseudo_count": 1,
        "cat_order": "target",
    }
    datasets = cf.get_data(config, Path("some/dir"))

    assert patched_load_data == [(3, Path("some/dir"))]
    _check_shapes(datasets)
    train_X = datasets[0][0]
    assert all(train_X[col].dtype == pl.Float64 for col in cf.CAT_COLS)


def test_get_data_target_encoding_requires_pseudo_count(patched_load_data):
    config = {"sample_every": 1, "cat_cutoff": 1, "cat_order": "target"}
    with pytest.raises(KeyError, match="cat_pseudo_count"):
        cf.get_data(config, Path("some/dir"))


@pytest.mark.parametrize("cat_order", ["tree", "node", "ohe", None])
def test_get_data_enum_encoding_without_pseudo_count(patched_load_data, cat_order):
    # cat_pseudo_count is only needed for target encoding
    config = {"sample_every": 1, "cat_cutoff": 1, "cat_order": cat_order}
    datasets = cf.get_data(config, Path("some/dir"))

    _check_shapes(datasets)
    train_X = datasets[0][0]
    assert all(isinstance(train_X[col].dtype, pl.Enum) for col in cf.CAT_COLS)


def test_get_data_enum_encoding(patched_load_data):
    config = {
        "sample_every": 1,
        "cat_cutoff": 1,
        "cat_pseudo_count": 1,
        "cat_order": "tree",
    }
    datasets = cf.get_data(config, Path("some/dir"))

    _check_shapes(datasets)
    train_X = datasets[0][0]
    assert all(isinstance(train_X[col].dtype, pl.Enum) for col in cf.CAT_COLS)
    # the null injected into C1 is filled as "missing" before encoding
    assert "missing" in train_X["C1"].to_list()
