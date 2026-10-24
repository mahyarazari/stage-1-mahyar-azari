"""Publish stage: the three final outputs in data/curated/.

  1. cases.csv                       the case-level dataset (one row per 311 case)
  2. summary_<level>_<metric>.csv    wide tables: one row per area x complaint type, one
                                     column per month (D8, D28). Metrics:
                                       median_days   median days to close
                                       cases         number of problems (duplicates once)
                                       rate          problems per `rate_per` residents
  3. quality_report.csv              per column: missing share, distinct values, range

How a cell is computed (D28 in docs/decisions.md):
  - Rows marked is_duplicate are left out: a duplicate group counts as one problem (D14).
  - Median: open cases count as "longer than any closed case" (infinite duration) instead of
    being dropped, so the median is not pulled down by cases that are still open. If more
    than half of a cell is open, the median is unknown and left empty.
  - A cell with fewer than `min_cases_per_cell` cases shows its count but no median or rate:
    a median of a handful of cases is noise, not a trend (D8).
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from data_factory.config import AggregateConfig
from data_factory.contract import Scope
from data_factory.enrichment import AREAS_FILE, ENRICHED_FILE
from data_factory.exceptions import MissingInputError
from data_factory.tables import read_csv_text, write_csv

log = logging.getLogger(__name__)

CASES_FILE = "cases.csv"
QUALITY_FILE = "quality_report.csv"
STATS_FILE = "publish_stats.json"
METRICS = ("median_days", "cases", "rate")
MEDIAN_DECIMALS = 2
RATE_DECIMALS = 3
NUMERIC_COLUMNS = ("days_to_close", "latitude", "longitude", "population")
SUMMARY_COLUMNS = ["borough", "modzcta", "complaint_type", "created_month",
                   "days_to_close", "is_open", "is_duplicate"]


@dataclass(frozen=True)
class PublishStats:
    cases_rows: int
    summary_rows: dict[str, int]           # rows of each summary table
    cells_hidden_small: dict[str, int]     # cells with too few cases for a median or rate
    cells_median_unknown: dict[str, int]   # cells where more than half of the cases are open
    cases_without_area: dict[str, int]     # problems left out of a level (no borough / no area)


def summary_file(level: str, metric: str) -> str:
    return f"summary_{level}_{metric}.csv"


def window_months(scope: Scope) -> list[str]:
    last = scope.window_end - pd.Timedelta(seconds=1)
    return [str(month) for month in pd.period_range(scope.window_start, last, freq="M")]


def _copy_atomically(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


# ---------------------------------------------------------------- summaries
def _median_with_open(durations: pd.Series) -> float:
    value = durations.median()
    return math.nan if math.isinf(value) else value


def summarise(cases: pd.DataFrame, level: str, population: dict[str, float], months: list[str],
              settings: AggregateConfig) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Wide tables for one level ('borough' or 'modzcta')."""
    usable = cases[~cases["is_duplicate"] & cases[level].notna()].copy()
    usable["duration"] = usable["days_to_close"].where(~usable["is_open"], math.inf)
    keys = [level, "complaint_type", "created_month"]
    grouped = usable.groupby(keys)
    long = pd.DataFrame({"cases": grouped.size(),
                         "median_days": grouped["duration"].agg(_median_with_open)})
    people = long.index.get_level_values(level).map(population)
    long["rate"] = long["cases"] / pd.Series(people, index=long.index) * settings.rate_per

    small = long["cases"] < settings.min_cases_per_cell
    unknown = long["median_days"].isna() & ~small
    long.loc[small, ["median_days", "rate"]] = math.nan
    long["median_days"] = long["median_days"].round(MEDIAN_DECIMALS)
    long["rate"] = long["rate"].round(RATE_DECIMALS)

    tables = {}
    for metric in METRICS:
        wide = long[metric].unstack("created_month").reindex(columns=months)
        if metric == "cases":
            wide = wide.fillna(0).astype("int64")
        wide.columns.name = None
        tables[metric] = wide.sort_index().reset_index()
    info = {"hidden": int(small.sum()), "unknown": int(unknown.sum()),
            "without_area": int((~cases["is_duplicate"] & cases[level].isna()).sum())}
    return tables, info


def _load_for_summary(path: Path) -> pd.DataFrame:
    cases = pd.read_csv(path, usecols=SUMMARY_COLUMNS, dtype="string",
                        keep_default_na=False, na_values=[""])
    cases["days_to_close"] = pd.to_numeric(cases["days_to_close"]).astype("float64")
    for flag in ("is_open", "is_duplicate"):
        cases[flag] = cases[flag].map({"True": True, "False": False}).astype(bool)
    return cases


def _population_by_level(areas: pd.DataFrame) -> dict[str, dict[str, float]]:
    people = pd.to_numeric(areas["population"]).astype("float64")
    by_area = dict(zip(areas["modzcta"], people.where(people > 0)))
    by_borough = people.where(people > 0).groupby(areas["borough"]).sum(min_count=1).to_dict()
    return {"modzcta": by_area, "borough": by_borough}


# ---------------------------------------------------------------- quality report
def quality_report(path: Path) -> pd.DataFrame:
    """Per column: missing share, distinct values and range. One column in memory at a time."""
    columns = list(pd.read_csv(path, nrows=0).columns)
    rows = []
    for column in columns:
        values = pd.read_csv(path, usecols=[column], dtype="string", keep_default_na=False,
                             na_values=[""])[column]
        present = values.dropna()
        if column in NUMERIC_COLUMNS:
            numbers = pd.to_numeric(present)
            low, high = (numbers.min(), numbers.max()) if len(numbers) else (None, None)
        else:
            low, high = (present.min(), present.max()) if len(present) else (None, None)
        top = present.value_counts().sort_index().sort_values(ascending=False, kind="stable")
        rows.append({
            "column": column,
            "rows": len(values),
            "missing": int(values.isna().sum()),
            "missing_pct": round(100 * values.isna().mean(), 4) if len(values) else 0.0,
            "distinct": int(present.nunique()),
            "min": low,
            "max": high,
            "most_common": top.index[0] if len(top) else None,
            "most_common_count": int(top.iloc[0]) if len(top) else 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage
def publish(work_dir: Path, curated_dir: Path, scope: Scope, settings: AggregateConfig) -> PublishStats:
    enriched, areas_path = work_dir / ENRICHED_FILE, work_dir / AREAS_FILE
    for path in (enriched, areas_path):
        if not path.exists():
            raise MissingInputError(f"{path} not found: run the enrich stage first")

    _copy_atomically(enriched, curated_dir / CASES_FILE)
    cases = _load_for_summary(enriched)
    population = _population_by_level(read_csv_text(areas_path))
    months = window_months(scope)

    summary_rows, hidden, unknown, without_area = {}, {}, {}, {}
    for level in settings.levels:
        tables, info = summarise(cases, level, population[level], months, settings)
        for metric, table in tables.items():
            name = summary_file(level, metric)
            summary_rows[name] = write_csv(curated_dir / name, table)
        hidden[level], unknown[level], without_area[level] = info["hidden"], info["unknown"], info["without_area"]

    write_csv(curated_dir / QUALITY_FILE, quality_report(curated_dir / CASES_FILE))

    stats = PublishStats(cases_rows=len(cases), summary_rows=summary_rows,
                         cells_hidden_small=hidden, cells_median_unknown=unknown,
                         cases_without_area=without_area)
    (work_dir / STATS_FILE).write_text(json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    log.info("publish: %d cases; summaries %s; small cells hidden %s",
             stats.cases_rows, summary_rows, hidden)
    return stats
