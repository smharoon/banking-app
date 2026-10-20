"""Generate one self-contained HTML EDA report per CSV batch."""
from __future__ import annotations

import base64
import html
import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

from app.database import CATEGORICAL, NUMERIC


def figure_to_html(fig) -> str:
    stream = io.BytesIO()
    fig.tight_layout()
    fig.savefig(stream, format="png", dpi=105, bbox_inches="tight")
    plt.close(fig)
    return '<img alt="EDA chart" src="data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode() + '">'


def table_html(df: pd.DataFrame, limit: int = 20) -> str:
    return df.head(limit).to_html(index=False, classes="data-table", border=0,
                                    na_rep="—", escape=True)


def bar_chart(series: pd.Series, title: str, color="#3869aa") -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    series.plot(kind="bar", ax=ax, color=color)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Count" if "rate" not in title.lower() else "Subscription rate")
    return figure_to_html(fig)


def cramers_v(left: pd.Series, right: pd.Series) -> float | None:
    contingency = pd.crosstab(left.fillna("(missing)"), right.fillna("(missing)"))
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    statistic = chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    minimum = min(contingency.shape) - 1
    return float(np.sqrt(statistic / (n * minimum))) if n > 0 and minimum > 0 else None


def build_report(df: pd.DataFrame, batch_id: int, filename: str, report_dir: Path) -> tuple[Path, dict]:
    """Analyse an in-memory COPY; never alter source rows stored in PostgreSQL."""
    report_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    missing_before = work.isna().sum()
    unknown_before = {col: int(work[col].astype(str).str.lower().eq("unknown").sum())
                      for col in CATEGORICAL}

    # All analysis operates on a separate in-memory copy of the input.
    missing_count = work.isna().sum()
    missing_percentage = (missing_count / len(work) * 100).round(2) if len(work) else missing_count.astype(float)
    total_missing = int(missing_count.sum())
    if total_missing == 0:
        missing_message = (
            "No NULL/missing values were detected in this batch. "
        )
    else:
        missing_message = (
            f"Detected {total_missing} NULL/missing values."
            "If treatment is required before model training, numeric fields could use "
            "training-set median imputation and categorical fields could use "
            "mode imputation or an explicit missing category. "
            "The original database records remain unchanged."
        )

    # Demonstrate normalization, one-hot encoding, and binning on analysis data.
    std = work[NUMERIC].std(ddof=0).replace(0, 1).fillna(1)
    norm = (work[NUMERIC] - work[NUMERIC].mean()) / std
    encoded = pd.get_dummies(work[CATEGORICAL].drop(columns=["y"]), dtype=int)
    age_groups = pd.cut(work["age"], bins=[0, 29, 39, 49, 59, 200],
                        labels=["18–29", "30–39", "40–49", "50–59", "60+"],
                        include_lowest=True)

    plots = []
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if work["age"].notna().any():
        work["age"].dropna().plot(kind="hist", bins=min(15, max(1, len(work))),
                                 ax=ax, color="#3869aa", edgecolor="white")
    else:
        ax.text(0.5, 0.5, "No non-null age values", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Univariate: client age distribution")
    ax.set_xlabel("Age")
    plots.append(figure_to_html(fig))
    plots.append(bar_chart(work["job"].value_counts().head(10), "Univariate: jobs"))
    plots.append(bar_chart(work["y"].value_counts().reindex(["no", "yes"], fill_value=0),
                           "Target distribution: term-deposit subscription"))

    numeric_corr = work[NUMERIC].corr()
    if len(work) >= 3 and numeric_corr.notna().any().any():
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(numeric_corr.to_numpy(dtype=float), cmap="coolwarm",
                          vmin=-1, vmax=1)
        ax.set_xticks(range(len(NUMERIC)), NUMERIC, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(NUMERIC)), NUMERIC, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
        ax.set_title("Numeric-feature Pearson correlations")
        plots.append(figure_to_html(fig))

    grouped = (work.assign(subscribed=(work["y"] == "yes").astype(int))
               .groupby("job")["subscribed"].agg(["mean", "size"])
               .query("size >= 1").sort_values("size", ascending=False).head(10))
    if not grouped.empty:
        plots.append(bar_chart(grouped["mean"], "Subscription rate by job"))
    bins_summary = pd.DataFrame({"age_group": age_groups.astype(str),
                                 "subscribed": (work["y"] == "yes").astype(int)})
    bins_summary = (bins_summary.groupby("age_group", sort=True)["subscribed"]
                    .agg(["count", "mean"]).reset_index())

    associations = []
    for col in ["job", "marital", "education", "contact", "poutcome", "housing", "loan"]:
        value = cramers_v(work[col], work["y"])
        associations.append({"Feature": col, "Cramér's V vs y": round(value, 4) if value is not None else None})

    # Exploratory feature importance ONLY; actual model fitting and 70/30
    # evaluation belong to the separate MLOps flow. Drop duration to avoid
    # pre-call target leakage. Treat pdays=999 as 'not previously contacted'.
    feature_note = "Not available: a meaningful feature-importance analysis needs >=20 rows and both target classes."
    if len(work) >= 20 and work["y"].nunique() == 2:
        sample = work.sample(n=min(10000, len(work)), random_state=42).copy()
        sample["previously_contacted"] = (sample["pdays"] != 999).astype(int)
        # The pdays=999 sentinel means no prior contact, not 999 elapsed days.
        # Work on a separate exploratory sample without imputing missing data.
        sample.loc[sample["pdays"] == 999, "pdays"] = 0
        features = sample.drop(columns=["y", "duration"])
        if features.isna().any().any():
            feature_note = (
                "Feature importance skipped because missing values are present. "
                "No values were imputed or changed in the original dataset."
            )
        else:
            x = pd.get_dummies(features, dtype=int)
            model = RandomForestClassifier(n_estimators=40, max_depth=8,
                                           random_state=42, n_jobs=1)
            model.fit(x, (sample["y"] == "yes").astype(int))
            importance = pd.Series(model.feature_importances_, index=x.columns)
            importance = importance.sort_values(ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 4))
            importance.sort_values().plot(kind="barh", ax=ax, color="#398178")
            ax.set_xlabel("Relative importance (exploratory only)")
            ax.set_title("Top features excluding call duration")
            plots.append(figure_to_html(fig))
            feature_note = "Exploratory Random Forest feature importance (not model-validation metrics). Call duration excluded."

    quality = {
        "rows": len(df),
        "missing_before": total_missing,
        "unknown_labels": int(sum(unknown_before.values())),
        "missing_values": total_missing,
        "imputation_performed": False,
        "duplicate_rows_in_file": int(df.duplicated().sum()),
        "encoded_feature_count": len(encoded.columns),
        "numeric_columns": len(NUMERIC),
        "categorical_columns": len(CATEGORICAL),
    }
    missing_summary = pd.DataFrame({
        "Feature": work.columns,
        "Missing Count": [int(missing_count[c]) for c in work.columns],
        "Missing Percentage (%)": [float(missing_percentage[c]) for c in work.columns],
    })
    data_types = pd.DataFrame({"Column": df.columns, "Data type": df.dtypes.astype(str).values})
    stat = df[NUMERIC].describe().round(3).T.reset_index(names="Feature")
    norm_preview = norm.head(5).round(3).reset_index(drop=True)
    unknown_summary = pd.DataFrame({"Feature": list(unknown_before),
                                    "'unknown' count": list(unknown_before.values())})
    sections = [
        ("Dataset summary", f"<p><b>Rows:</b> {len(df):,} &nbsp; <b>Columns:</b> {len(df.columns)} &nbsp; <b>Duplicate rows:</b> {quality['duplicate_rows_in_file']}</p>" + table_html(stat)),
        ("Data types", table_html(data_types, 25)),
        ("Missing value analysis", "<p>" + html.escape(missing_message) + "</p>" + table_html(missing_summary, 25)),
        ("Categorical unknown values", "<p>The UCI dataset often represents missing categories as the literal string <code>unknown</code>.</p>" + table_html(unknown_summary)),
        ("Normalization (z-score sample; analysis only, original values unchanged)", table_html(norm_preview)),
        ("One-hot encoding", f"<p>Encoded {len(encoded.columns)} categorical indicator columns. First 12 names: {html.escape(', '.join(encoded.columns[:12]))}</p>"),
        ("Age binning", table_html(bins_summary)),
        ("Categorical association with target", "<p>Cramér's V values range from 0 (no measured association) to 1 (strong association). Values are not defined for single-class samples.</p>" + table_html(pd.DataFrame(associations))),
        ("Exploratory feature importance", "<p>" + html.escape(feature_note) + "</p>"),
        ("Plots: univariate and bivariate", "".join(f'<div class="chart">{p}</div>' for p in plots)),
    ]
    css = """
    body{font-family:Arial,Helvetica,sans-serif;line-height:1.5;color:#182735;max-width:1100px;margin:35px auto;padding:0 22px;background:#f7fafc}
    header,section{background:white;border:1px solid #dce5ed;border-radius:10px;padding:18px 22px;margin-bottom:18px}
    h1{margin:0 0 8px}h2{margin-top:0;color:#155985}p{color:#334a5c}
    .data-table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow-x:auto}
    .data-table td,.data-table th{padding:6px 10px;border-bottom:1px solid #e4ebf1;text-align:left;white-space:nowrap}
    .data-table th{background:#eef5fa} .chart{margin:14px 0;text-align:center} img{max-width:100%;height:auto}
    .meta{color:#647987;font-size:13px}code{background:#f0f3f6;padding:2px 4px}
    """
    content = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
               f"<title>Bank Marketing EDA - Batch {batch_id}</title><style>{css}</style></head><body>",
               "<header><h1>Bank Marketing — DataOps Report</h1>",
               f"<p class='meta'>Batch ID: {batch_id} | File: {html.escape(filename)} | Generated UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}</p>",
               "<p>Business objective: understand clients and factors associated with subscription to a term deposit.</p></header>"]
    for heading, content_html in sections:
        content.append(f"<section><h2>{html.escape(heading)}</h2>{content_html}</section>")
    content.append("<footer class='meta'>Moro, Cortez and Rita (2014), A Data-Driven Approach to Predict the Success of Bank Telemarketing, Decision Support Systems. DOI: 10.1016/j.dss.2014.03.001.</footer></body></html>")
    output = report_dir / f"batch_{batch_id:04d}_report.html"
    output.write_text("\n".join(content), encoding="utf-8")
    return output, quality
