import json
import re

import polars as pl
import polars.selectors as cs

import wandb


def get_project_sweeps(
    entity: str = "seanv507",
) -> dict[str, list[dict]]:
    """
    Return every accessible project under `entity` and its sweep IDs.

    Projects without sweeps are included with an empty list.
    """
    api = wandb.Api(timeout=60)

    return {
        project.name: [
            {
                "sweep_id": sweep.id,
                "sweep_config": sweep.config,
                "sweep_runs": [
                    {"run_id": run.id, "run_summary_metrics": run.summary_metrics}
                    for run in sweep.runs
                ],
            }
            for sweep in project.sweeps(per_page=100)
        ]
        for project in api.projects(entity=entity, per_page=200)
    }


def make_sweeps_df(project_sweeps: dict[str, list[dict]]):
    project_sweeps_df = (
        pl.DataFrame(
            {"project": project_sweeps.keys(), "sweeps": project_sweeps.values()},
            strict=False,
        )
        .explode("sweeps")
        .unnest("sweeps")
        .explode("sweep_runs")
        .unnest("sweep_runs")
        .with_columns(
            pl.col("run_id"),
            (pl.col("sweep_config").struct["parameters"]),
            training_logloss=pl.col("run_summary_metrics").struct[
                "validation_0-logloss.min"
            ],
            validation_logloss=pl.col("run_summary_metrics").struct[
                "validation_1-logloss.min"
            ],
        )
        .unnest("parameters")
        .sort("validation_logloss")
    )
    return project_sweeps_df


def get_project_runs(
    entity: str = "seanv507",
    project_name: str | None = None,
    filters: dict | None = None,
    max_runs: int | None = None,
) -> dict[str, list[dict]]:
    """
    Return every accessible project under `entity` and its runs, whether or
    not they belong to a sweep.

    `project_name` is an optional regex (matched with `re.search`) used to
    select only projects whose name matches. Projects without runs are
    included with an empty list. Runs outside a sweep have `sweep_id=None`.
    """
    api = wandb.Api(timeout=60)
    pattern = re.compile(project_name) if project_name else None
    # system_metrics system.gpu.0.memoryAllocatedBytes': 198049792,
    run_count = 0
    project_runs = {}
    for project in api.projects(entity=entity, per_page=200):
        if not (pattern is None or pattern.search(project.name)):
            continue
        runs = []
        for run in api.runs(f"{entity}/{project.name}", per_page=200, filters=filters):
            fi_ref = run.summary.get("Feature Importance_table")
            if fi_ref:
                file = run.file(fi_ref["path"]).download(replace=True)
                with open(file.name) as f:
                    contents = json.load(f)
                    fi = [
                        {"feature": row[0], "value": row[1]} for row in contents["data"]
                    ]
                    # fi = dict(contents["data"])
            else:
                fi = None
            run_d = {
                "run_id": run.id,
                "sweep_id": run.sweep.id if run.sweep else None,
                "state": run.state,
                "name": run.name,
                "created_at": run.created_at,
                "job_type": run.job_type,
                "tags": run.tags,
                "notes": run.notes,
                "group": run.group,
                "config": dict(run.config),
                "metadata": dict(run.metadata) if run.metadata else None,
                "system_metrics": dict(run.system_metrics)
                if run.system_metrics
                else None,
                "summary_metrics": dict(run.summary_metrics)
                if run.summary_metrics
                else None,
                "feature_importance": fi,
            }
            runs.append(run_d)
            run_count += 1
            if max_runs and run_count >= max_runs:
                break
        project_runs[project.name] = runs
        if max_runs and run_count >= max_runs:
            break

    return project_runs


def make_runs_df(project_runs: dict[str, list[dict]]):
    run_dfs = []
    for project, runs in project_runs.items():
        run_df = (
            pl.DataFrame(
                runs,
                strict=False,
            )
            # schema_overrides={"feature_importance": pl.List(pl.Struct({"feature": pl.String, "importance": pl.Float64}))}
        ).insert_column(0, pl.lit(project).alias("project"))
        run_dfs.append(run_df)
    runs = (
        pl.concat(run_dfs, how="vertical_relaxed").drop_nulls(
            subset="run_id"
        )  # projects with no runs
    )
    runs = unnest_full(runs, exclude="feature_importance")
    return runs


def unnest_full(df, separator=".", exclude=None):
    while True:
        df_flatten = df.unnest(cs.struct().exclude(exclude), separator=separator)
        if df_flatten.columns == df.columns:
            break
        df = df_flatten
    return df_flatten


def cast_datetimes(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.from_epoch(cs.ends_with("timestamp")),
        cs.ends_with("created_at").str.to_datetime("%+"),
    )
    return df


def convert_int_double_strs(df):
    int_convertible = convertible(df, pl.Int64)
    float_convertible = [
        col for col in convertible(df, pl.Float64) if col not in int_convertible
    ]
    df = df.with_columns(
        pl.col(int_convertible).cast(pl.Int64),
        pl.col(float_convertible).cast(pl.Float64),
    )
    return df


def convertible(df: pl.DataFrame, dtype) -> list[str]:
    str_cols = df.select(pl.col(pl.String)).columns
    res = df.select(
        (
            pl.col(c).cast(dtype, strict=False).null_count() == pl.col(c).null_count()
        ).alias(c)
        for c in str_cols
    )
    return [k for k, v in res.row(0, named=True).items() if v]


def identify_variation(runs_df: pl.DataFrame):
    runs_unique = runs_df.select(pl.all().value_counts(sort=True).implode())
    return runs_unique


def one_at_a_time(grid, params):
    """One row per 1-D slice: all params but `varying` fixed, `varying` takes >1 value."""
    #     Columns: the varying column is set to null (it's the free axis), and the other parameter columns hold the values held fixed.
    # values dtype: it's cast to Float64 so the per-parameter frames can be stacked, because reg_lambda is int and min_split_loss is float.
    # Slices can share a run. For example, the run at (25600, 400, 5.0, 1.0) belongs to the cat_reg_lambda, min_split_loss and min_child_weight slices.
    #  Repeated settings: settings with len 2, like (0, 10, …) and (12800, 100, …), count once in values because of unique(). Their runs show up when you join back, as below.

    metric = "validation log loss"  # "summary_metrics.validation_1-logloss.min"
    return pl.concat(
        [
            grid.group_by(p for p in params if p != v)
            .agg(
                pl.col(v).unique().sort().cast(pl.Float64).alias("values"),
                pl.col("run_id"),
                pl.col(metric),
            )
            .filter(pl.col("values").list.len() > 1)
            .with_columns(pl.lit(v).alias("varying"), pl.lit(None).alias(v))
            .select("varying", *params, "values", "run_id", metric)
            for v in params
        ],
        how="vertical_relaxed",
    ).sort("varying", pl.col("values").list.len(), descending=[False, True])


def replace_strings(name: str):
    replacements = {
        "config.": "",
        "learner.gradient_booster.tree_train_param.": "",
        "learner.learner_model_param.": "",
        "summary_metrics.validation_1-logloss.min": "validation log loss",
    }

    for pattern, repl in replacements.items():
        name = re.sub(pattern, repl, name)
    return name
