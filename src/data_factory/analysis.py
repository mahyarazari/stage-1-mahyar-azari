"""Analysis helpers for the notebook: evidence checks, EDA tables and plots.

The notebook only calls these functions and explains the results; the logic lives here so
it can be read, reused and imported (delivery guide, rule 6). Plots use matplotlib and
return the figure; labels are in English because matplotlib does not shape Persian text.
"""
from __future__ import annotations

import math
import tempfile
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_factory.contract import Scope
from data_factory.exceptions import QuarantineThresholdError
from data_factory.raw_store import RawPageStore
from data_factory.references import BoroughReference
from data_factory.validation import QUARANTINE_FILE, validate_snapshot

DAYS_PER_YEAR = 365
OUTLIER_IQR_FACTOR = 3.0          # "extreme" outliers: beyond Q3 + 3 x IQR
CORRUPTED_RECORDS = 10            # records broken on purpose in each quarantine demo
BAD_DATE_FORMAT = "%m/%d/%Y"      # a date format the contract does not accept

CASE_COLUMNS = {
    "unique_key": "string", "complaint_type": "category", "borough": "category",
    "created_month": "category", "days_to_close": "float64", "is_open": "string",
    "is_duplicate": "string", "incident_address": "string", "incident_zip": "string",
    "open_data_channel_type": "category", "population": "float64", "modzcta": "string",
}


# ---------------------------------------------------------------- loading
def load_cases(path: Path) -> pd.DataFrame:
    """The columns the analysis needs, with compact types."""
    cases = pd.read_csv(path, usecols=list(CASE_COLUMNS), dtype=CASE_COLUMNS,
                        keep_default_na=False, na_values=[""])
    for flag in ("is_open", "is_duplicate"):
        cases[flag] = cases[flag] == "True"
    return cases


def problems(cases: pd.DataFrame) -> pd.DataFrame:
    """One row per problem: duplicate rows left out (D14)."""
    return cases[~cases["is_duplicate"]]


# ---------------------------------------------------------------- evidence
def _peak_reading(stores: list[RawPageStore], max_pages: int | None = None) -> tuple[int, int, int]:
    """Read pages one at a time; return (pages read, records read, peak traced bytes)."""
    pages = records = 0
    tracemalloc.start()
    try:
        for store in stores:
            for page in store.read_pages():
                pages += 1
                records += len(page)
                if max_pages is not None and pages >= max_pages:
                    break
            if max_pages is not None and pages >= max_pages:
                break
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return pages, records, peak


def raw_memory_profile(stores: list[RawPageStore]) -> dict[str, float]:
    """Peak memory while reading ONE page, and while reading ALL pages, one at a time.

    If the whole raw data were held in memory, the second peak would grow with the number
    of pages. Reading page by page, both peaks are about the same.
    """
    raw_bytes = sum(p.stat().st_size for store in stores for p in store.page_files())
    _, _, one_page_peak = _peak_reading(stores, max_pages=1)
    pages, records, all_pages_peak = _peak_reading(stores)
    return {"pages": pages, "records": records,
            "raw_mb_on_disk": round(raw_bytes / 1e6, 1),
            "peak_mb_reading_one_page": round(one_page_peak / 1e6, 1),
            "peak_mb_reading_all_pages": round(all_pages_peak / 1e6, 1),
            "all_pages_peak_vs_one_page": round(all_pages_peak / one_page_peak, 2) if one_page_peak else None,
            "estimated_mb_if_all_pages_were_held": round(pages * one_page_peak / 1e6, 1)}


def _validate_copy(records: list[dict], scope: Scope, placeholders: tuple[str, ...],
                   max_ratio: float, folder: Path, fetched_at: str):
    store = RawPageStore(folder / "raw")
    store.write_page(1, records)
    store.mark_complete({"fetched_at_utc": fetched_at})
    stats = validate_snapshot([store], folder / "work", folder / "quarantine", scope, placeholders, max_ratio)
    quarantined = pd.read_csv(folder / "quarantine" / QUARANTINE_FILE, dtype=str)
    return stats, quarantined


def quarantine_demo(store: RawPageStore, scope: Scope, placeholders: tuple[str, ...],
                    max_ratio: float) -> dict[str, object]:
    """Break a copy of the first real page on purpose and show what the contract does.

    1. A few records broken (closed before created; unreadable date): they must be
       quarantined with the right reason, and every other record must pass.
    2. Every record broken (the source changed its date format): the run must stop.
    The real snapshot is never touched; everything happens in a temporary folder.
    """
    page = next(store.read_pages())
    fetched_at = store.completion_info()["fetched_at_utc"]
    broken = [dict(r) for r in page]
    closed_too_early, bad_date = [], []
    for record in broken[:CORRUPTED_RECORDS]:
        record["closed_date"] = "2023-01-01T00:00:00.000"
        closed_too_early.append(record["unique_key"])
    for record in broken[CORRUPTED_RECORDS:2 * CORRUPTED_RECORDS]:
        stamp = pd.Timestamp(record["created_date"])
        record["created_date"] = stamp.strftime(BAD_DATE_FORMAT)
        bad_date.append(record["unique_key"])

    with tempfile.TemporaryDirectory() as folder:
        stats, quarantined = _validate_copy(broken, scope, placeholders, 1.0, Path(folder), fetched_at)
    by_key = dict(zip(quarantined["unique_key"], quarantined["reason_codes"]))
    small = {
        "records_in_copy": stats.raw_records,
        "quarantined": stats.quarantined_records,
        "exactly_the_broken_ones": set(by_key) == set(closed_too_early + bad_date),
        "closed_before_created_caught": all(by_key.get(k) == "CLOSED_BEFORE_CREATED" for k in closed_too_early),
        "unparseable_date_caught": all(by_key.get(k) == "UNPARSEABLE_DATE" for k in bad_date),
        "reasons": {k: v for k, v in stats.reasons.items() if v},
    }

    everything = [dict(r) for r in page]
    for record in everything:
        record["created_date"] = pd.Timestamp(record["created_date"]).strftime(BAD_DATE_FORMAT)
    with tempfile.TemporaryDirectory() as folder:
        try:
            _validate_copy(everything, scope, placeholders, max_ratio, Path(folder), fetched_at)
            stopped, message, published = False, "", True
        except QuarantineThresholdError as error:
            stopped, message = True, str(error).split(";")[0]
            published = (Path(folder) / "work" / "validated.csv").exists()
    return {"small_corruption": small,
            "wide_corruption": {"run_stopped": stopped, "message": message,
                                "valid_file_published": published}}


def borough_rule_comparison(cases: pd.DataFrame, boroughs: BoroughReference) -> dict[str, object]:
    """311 borough against the zip's borough, with the naive rule and with the legal rule (D7)."""
    checked = cases[cases["incident_zip"].isin(boroughs.zip_land) & cases["borough"].notna()]
    largest = {z: max(land.items(), key=lambda item: (item[1], item[0]))[0]
               for z, land in boroughs.zip_land.items()}
    naive = checked["incident_zip"].map(largest) != checked["borough"].astype(str)
    legal = [b not in boroughs.zip_boroughs.get(z, ()) for z, b in
             zip(checked["incident_zip"], checked["borough"].astype(str))]
    by_zip = (checked[naive].groupby(["incident_zip", "borough"], observed=True).size()
              .rename("rows").reset_index().sort_values("rows", ascending=False))
    by_zip["largest_area_rule_says"] = by_zip["incident_zip"].map(largest)
    by_zip["zip_is_split"] = by_zip["incident_zip"].isin(boroughs.split_zips)
    return {"rows_checked": len(checked), "largest_area_rule_disagreements": int(naive.sum()),
            "legal_rule_disagreements": int(sum(legal)), "disagreements_by_zip": by_zip}


def cell_size_distribution(summary_cases: pd.DataFrame, thresholds=(5, 10, 15, 20, 30, 50)) -> pd.DataFrame:
    """How many neighbourhood cells would keep a median with each minimum (D8)."""
    values = summary_cases.iloc[:, 2:].to_numpy(dtype=float).ravel()
    values = values[values > 0]
    return pd.DataFrame({"min_cases": thresholds,
                         "cells_kept": [int((values >= k).sum()) for k in thresholds],
                         "share_kept_pct": [round(100 * (values >= k).mean(), 1) for k in thresholds]})


# ---------------------------------------------------------------- EDA tables
def duration_summary(cases: pd.DataFrame) -> pd.DataFrame:
    """Days to close per complaint type (closed problems), plus the share still open."""
    data = problems(cases)
    closed = data[~data["is_open"]].groupby("complaint_type", observed=True)["days_to_close"]
    table = pd.DataFrame({
        "problems": data.groupby("complaint_type", observed=True).size(),
        "open_pct": (100 * data.groupby("complaint_type", observed=True)["is_open"].mean()).round(2),
        "median": closed.median().round(2), "p25": closed.quantile(0.25).round(2),
        "p75": closed.quantile(0.75).round(2), "p90": closed.quantile(0.90).round(2),
        "mean": closed.mean().round(2), "max": closed.max().round(1),
    })
    return table.sort_values("median")


def outlier_summary(cases: pd.DataFrame) -> pd.DataFrame:
    """Extreme durations per type: beyond Q3 + 3 x IQR, and beyond one year."""
    rows = []
    closed = problems(cases)
    closed = closed[~closed["is_open"]]
    for kind, days in closed.groupby("complaint_type", observed=True)["days_to_close"]:
        q1, q3 = days.quantile([0.25, 0.75])
        fence = q3 + OUTLIER_IQR_FACTOR * (q3 - q1)
        rows.append({"complaint_type": kind, "fence_days": round(fence, 1),
                     "beyond_fence": int((days > fence).sum()),
                     "beyond_fence_pct": round(100 * (days > fence).mean(), 2),
                     "over_one_year": int((days > DAYS_PER_YEAR).sum()),
                     "zero_days": int((days == 0).sum())})
    return pd.DataFrame(rows).set_index("complaint_type")


def _median_open_as_slowest(days: pd.Series, is_open: pd.Series) -> float:
    value = days.where(~is_open, math.inf).median()
    return math.nan if math.isinf(value) else value


def heat_versus_others(cases: pd.DataFrame, heat_type: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per month: heat problems created, and the city-wide median days of the other types."""
    data = problems(cases)
    heat = data[data["complaint_type"] == heat_type].groupby("created_month", observed=True).size()
    others = data[data["complaint_type"] != heat_type]
    median = others.groupby("created_month", observed=True).apply(
        lambda g: _median_open_as_slowest(g["days_to_close"], g["is_open"]), include_groups=False)
    monthly = pd.DataFrame({"heat_problems": heat, "other_types_median_days": median}).sort_index()
    correlation = {
        "pearson": round(float(monthly["heat_problems"].corr(monthly["other_types_median_days"])), 3),
        # Spearman = Pearson on ranks (computed here so scipy is not needed)
        "spearman": round(float(monthly["heat_problems"].rank().corr(
            monthly["other_types_median_days"].rank())), 3),
        "months": len(monthly),
    }
    return monthly, correlation


def year_over_year(summary_median: pd.DataFrame) -> pd.DataFrame:
    """Same months, one year apart: mean of monthly medians in each year, and the change."""
    months = [c for c in summary_median.columns[2:]]
    pairs = [(m, f"{int(m[:4]) + 1}{m[4:]}") for m in months if f"{int(m[:4]) + 1}{m[4:]}" in months]
    first = [a for a, _ in pairs]
    second = [b for _, b in pairs]
    values = summary_median.set_index(list(summary_median.columns[:2]))
    table = pd.DataFrame({
        f"{first[0]}..{first[-1]}": values[first].astype(float).mean(axis=1).round(2),
        f"{second[0]}..{second[-1]}": values[second].astype(float).mean(axis=1).round(2),
    })
    table["change_pct"] = (100 * (table.iloc[:, 1] - table.iloc[:, 0]) / table.iloc[:, 0]).round(1)
    return table


def concentration(cases: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """How unevenly problems are spread over addresses (cumulative share curve)."""
    counts = problems(cases)["incident_address"].value_counts()
    curve = pd.DataFrame({
        "address_share_pct": 100 * np.arange(1, len(counts) + 1) / len(counts),
        "problem_share_pct": 100 * counts.cumsum().to_numpy() / counts.sum(),
    })
    top_1pct = max(1, int(round(0.01 * len(counts))))
    facts = {"addresses": len(counts), "problems": int(counts.sum()),
             "top_address": counts.index[0], "top_address_problems": int(counts.iloc[0]),
             "top_1pct_addresses": top_1pct,
             "top_1pct_share_pct": round(float(100 * counts.iloc[:top_1pct].sum() / counts.sum()), 1),
             "addresses_for_half_of_problems_pct":
                 round(float(curve.loc[curve["problem_share_pct"] >= 50, "address_share_pct"].iloc[0]), 1)}
    return curve, facts


def median_by_borough_and_type(cases: pd.DataFrame) -> pd.DataFrame:
    data = problems(cases).dropna(subset=["borough"])
    table = data.groupby(["borough", "complaint_type"], observed=True).apply(
        lambda g: _median_open_as_slowest(g["days_to_close"], g["is_open"]), include_groups=False)
    return table.unstack("complaint_type").round(2)


def channel_comparison(cases: pd.DataFrame) -> pd.DataFrame:
    data = problems(cases)
    table = data.groupby(["complaint_type", "open_data_channel_type"], observed=True).apply(
        lambda g: _median_open_as_slowest(g["days_to_close"], g["is_open"]), include_groups=False)
    return table.unstack("open_data_channel_type").round(2)


# ---------------------------------------------------------------- plots
def plot_duration_distribution(cases: pd.DataFrame):
    data = problems(cases)
    data = data[~data["is_open"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(-2, 3, 60)
    order = data.groupby("complaint_type", observed=True)["days_to_close"].median().sort_values().index
    for kind in order:
        days = data.loc[data["complaint_type"] == kind, "days_to_close"].clip(lower=0.01)
        ax.hist(days, bins=bins, histtype="step", linewidth=1.6, label=f"{kind} (median {days.median():.1f} d)")
    ax.set_xscale("log")
    ax.set_xlabel("days to close (log scale)")
    ax.set_ylabel("problems")
    ax.set_title("Distribution of days to close, by complaint type")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_heat_versus_others(monthly: pd.DataFrame):
    fig, left = plt.subplots(figsize=(11, 4.5))
    x = range(len(monthly))
    left.bar(x, monthly["heat_problems"], color="#f4a261", label="heat problems created")
    left.set_ylabel("HEAT/HOT WATER problems created")
    right = left.twinx()
    right.plot(x, monthly["other_types_median_days"], color="#264653", marker="o", linewidth=2,
               label="median days, other five types")
    right.set_ylabel("median days to close, other types")
    left.set_xticks(list(x))
    left.set_xticklabels(monthly.index.astype(str), rotation=60, fontsize=8)
    left.set_title("Heating season vs. speed of the other complaint types")
    handles = left.get_legend_handles_labels()[0] + right.get_legend_handles_labels()[0]
    left.legend(handles, [h.get_label() for h in handles], loc="upper center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_year_over_year(yoy: pd.DataFrame):
    change = yoy["change_pct"].unstack(level=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    change.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("change of median days to close (%)\nsame months, one year later")
    ax.set_xlabel("")
    ax.set_title("Better (below 0) or worse (above 0)? Year-over-year change by borough")
    ax.legend(fontsize=7, ncols=3)
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def plot_concentration(curve: pd.DataFrame, facts: dict[str, object]):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["address_share_pct"], curve["problem_share_pct"], linewidth=2, label="actual")
    ax.plot([0, 100], [0, 100], linestyle="--", color="grey", label="perfectly even")
    ax.axvline(1, color="#e76f51", linewidth=1)
    ax.annotate(f"top 1% of addresses:\n{facts['top_1pct_share_pct']}% of problems",
                xy=(1, facts["top_1pct_share_pct"]), xytext=(15, facts["top_1pct_share_pct"] - 15),
                arrowprops={"arrowstyle": "->"}, fontsize=9)
    ax.set_xlabel("addresses, most complained-about first (%)")
    ax.set_ylabel("share of all problems (%)")
    ax.set_title("How concentrated are problems on a few buildings?")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_borough_type_heatmap(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(table.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    for (i, j), value in np.ndenumerate(table.to_numpy(dtype=float)):
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="median days to close")
    ax.set_title("Median days to close, borough x complaint type (whole window)")
    fig.tight_layout()
    return fig
