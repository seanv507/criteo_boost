I’ll check the current Report/Table workflow and map it to this sweep’s exact config and metric keys, so the example is directly runnable.

The cleanest approach is to compute the cross-run statistics with pandas, log the result as a `wandb.Table` in one analysis run, and display that table in a Report. Reports do not automatically add min/max/mean rows to a sweep’s runs table.

```python
import pandas as pd
import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY = "seanv507"
PROJECT = "criteo_boost_sweep"
SWEEP_ID = "s15xvple"
TABLE_KEY = "sweep_column_statistics"

# 1. Read the 45 sweep runs.
api = wandb.Api()
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")

rows = []
for run in sweep.runs:
    summary = dict(run.summary)
    rows.append({
        "learning_rate": run.config.get("learning_rate"),
        "max_depth": run.config.get("max_depth"),
        "subsample": run.config.get("subsample"),
        # This sweep stores the objective's minimum under this summary key.
        "validation_1-logloss": summary.get(
            "validation_1-logloss.min"
        ),
    })

# 2. Calculate one row per column.
values = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

stats = (
    values.agg(["min", "max", "mean"])
    .T
    .reset_index()
    .rename(columns={"index": "column"})
)

# 3. Log the derived table to a dedicated analysis run.
with wandb.init(
    entity=ENTITY,
    project=PROJECT,
    name=f"{SWEEP_ID}-column-statistics",
    job_type="analysis",
) as analysis_run:
    analysis_run.log({
        TABLE_KEY: wandb.Table(dataframe=stats)
    })
    analysis_run_id = analysis_run.id

# 4. Create a Report containing the table.
report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="Criteo boost sweep statistics",
    description=f"Column statistics across all runs in sweep {SWEEP_ID}.",
    blocks=[
        wr.H2("Sweep column statistics"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    filters=[
                        wr.Metric("name").isin([analysis_run_id])
                    ],
                )
            ],
            panels=[
                wr.WeavePanelSummaryTable(
                    table_name=TABLE_KEY,
                    layout=wr.Layout(w=24, h=12),
                )
            ],
        ),
    ],
)

report.save()
print(report.url)
```

This creates one clearly labeled **analysis run** outside the sweep and a native, sortable W&B Table in the Report. Because the sweep is finished, the result is complete; if runs changed later, rerun the script to refresh the statistics.

References: [W&B Table](https://docs.wandb.ai/models/ref/python/data-types/table), [Report summary-table panel](https://docs.wandb.ai/models/ref/wandb_workspaces/reports#class-weavepanelsummarytable).



I’ll quantify each hyperparameter’s effect across the complete 45-run factorial grid, separate main effects from interactions, and then add a concise evidence-backed summary to this report.

The grid has exactly one run per hyperparameter combination, so a saturated ANOVA has no residual degrees of freedom. I’m using the balanced-grid sums of squares instead; in this complete factorial design, those effects are orthogonal and still give an exact variance decomposition.

The result is clear: `subsample` dominates the observed score variation, `learning_rate` is second, and `max_depth` is effectively inert over 50–70. I’ll add the quantified conclusion and the one-run-per-cell caveat to the existing draft without publishing it.

Added and verified a new **“Hyperparameter contribution to validation loss”** section in the existing draft report.

### Main finding

- **Subsample:** largest contribution—**60.7%** of observed loss variation. Moving from 0.3 to 1.0 improved mean logloss by **0.003547**.
- **Learning rate:** **38.3%**. The best average setting was **0.1**.
- **Max depth:** effectively negligible—**0.0004%** across depths 50–70.
- **Interactions:** about **1.0%**, mostly learning rate × subsample.

The best run used `learning_rate=0.1`, `max_depth=60`, and `subsample=1.0`, with validation logloss **0.444352**.

[Open the updated report](https://wandb.ai/seanv507/criteo_boost_sweep/reports/Criteo_boost_sweep--VmlldzoxNzkzOTM4OA==). It remains a draft and was not published.


