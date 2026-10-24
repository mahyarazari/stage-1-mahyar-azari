"""Enrich stage: add population and check/complete the borough of every case.

Implements D5-D7 and D10 of docs/decisions.md:
  - D6   zip aliases (e.g. 11249 -> 11211) are applied before the population lookup only;
         the original zip is kept, the zip actually used is in `join_zip`
  - D5   population of the case's MODZCTA area; area with no or zero population -> missing
  - D10  a record without population is kept and labelled (population_status), never dropped
  - D7   the 311 borough is the legal borough and is kept; the Census zip->county table is
         only a check: "is the 311 borough one of the boroughs this zip touches?"
  - D11  a missing borough is filled from the zip only when the zip lies in one borough

The join is a lookup (one value per key), so it cannot add or remove rows; the row count
before and after is still counted and checked, as the problem statement asks.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from data_factory.cleaning import CLEAN_COLUMNS, CLEANED_FILE
from data_factory.exceptions import DataValidationError, MissingInputError
from data_factory.references import BoroughReference, PopulationReference, area_boroughs
from data_factory.tables import AtomicCsvWriter, read_csv_text, write_csv

log = logging.getLogger(__name__)

ENRICHED_FILE = "enriched.csv"
AREAS_FILE = "areas.csv"
STATS_FILE = "enrich_stats.json"
TOP_UNMATCHED = 20

ENRICH_COLUMNS = CLEAN_COLUMNS + [
    "join_zip", "modzcta", "population", "population_status", "borough_source", "borough_check",
]

# population_status values
MATCHED = "matched"
ZIP_MISSING = "zip_missing"
ZIP_NOT_IN_REFERENCE = "zip_not_in_reference"
NO_POPULATION = "no_population"

# borough_check values
CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
NOT_CHECKABLE = "not_checkable"


@dataclass(frozen=True)
class EnrichStats:
    rows_before_join: int
    rows_after_join: int
    alias_applied: int
    population_status: dict[str, int]
    borough_source: dict[str, int]
    borough_check: dict[str, int]
    unmatched_zips: dict[str, int]          # most frequent zips with no population area
    inconsistent_pairs: dict[str, int]      # "zip|311 borough" -> rows
    reference: dict[str, object]


def enrich_chunk(cases: pd.DataFrame, population: PopulationReference,
                 boroughs: BoroughReference, aliases: dict[str, str]) -> pd.DataFrame:
    out = cases.copy()
    zip_code = cases["incident_zip"]
    join_zip = zip_code.replace(aliases)
    area = join_zip.map(population.zip_to_modzcta).astype("string")
    people = pd.to_numeric(area.map(population.modzcta_population), errors="coerce")

    status = pd.Series(MATCHED, index=cases.index, dtype="string")
    status = status.mask(people.isna() | (people <= 0), NO_POPULATION)
    status = status.mask(area.isna(), ZIP_NOT_IN_REFERENCE)
    status = status.mask(zip_code.isna(), ZIP_MISSING)

    out["join_zip"] = join_zip
    out["modzcta"] = area
    out["population"] = people.where(people > 0)
    out["population_status"] = status

    # Borough: check the 311 value against the original zip (the 2020 Census file knows 11249).
    touched = zip_code.map(boroughs.zip_boroughs)
    reported = cases["borough"]
    checkable = reported.notna() & touched.notna()
    agrees = pd.Series([isinstance(t, tuple) and isinstance(b, str) and b in t
                        for b, t in zip(reported, touched)], index=cases.index)
    check = pd.Series(NOT_CHECKABLE, index=cases.index, dtype="string")
    check = check.mask(checkable & agrees, CONSISTENT).mask(checkable & ~agrees, INCONSISTENT)

    single = touched.map(lambda t: t[0] if isinstance(t, tuple) and len(t) == 1 else None).astype("string")
    fill = reported.isna() & single.notna()
    out["borough"] = reported.astype("string").mask(fill, single)
    source = pd.Series(pd.NA, index=cases.index, dtype="string")
    out["borough_source"] = source.mask(reported.notna(), "311").mask(fill, "zip")
    out["borough_check"] = check
    return out


def enrich_dataset(work_dir: Path, population: PopulationReference, boroughs: BoroughReference,
                   aliases: dict[str, str], chunk_rows: int,
                   reference_meta: dict[str, object] | None = None) -> EnrichStats:
    source = work_dir / CLEANED_FILE
    if not source.exists():
        raise MissingInputError(f"{source} not found: run the clean stage first")

    rows_before = rows_after = alias_applied = 0
    counters = {name: Counter() for name in ("population_status", "borough_source", "borough_check")}
    unmatched, inconsistent = Counter(), Counter()

    with AtomicCsvWriter(work_dir / ENRICHED_FILE, ENRICH_COLUMNS) as writer:
        for cases in read_csv_text(source, chunksize=chunk_rows):
            enriched = enrich_chunk(cases, population, boroughs, aliases)
            if len(enriched) != len(cases):
                raise DataValidationError("the population join changed the number of rows")
            rows_before += len(cases)
            rows_after += len(enriched)
            alias_applied += int((enriched["join_zip"] != cases["incident_zip"]).fillna(False).sum())
            for name, counter in counters.items():
                counter.update(enriched[name].fillna("missing").tolist())
            missing_area = enriched["population_status"] == ZIP_NOT_IN_REFERENCE
            unmatched.update(enriched.loc[missing_area, "incident_zip"].tolist())
            bad = enriched["borough_check"] == INCONSISTENT
            inconsistent.update((enriched.loc[bad, "incident_zip"] + "|" + enriched.loc[bad, "borough"]).tolist())
            writer.append(enriched)
        writer.commit()

    areas = pd.DataFrame(
        [(code, population.modzcta_population[code], borough)
         for code, borough in area_boroughs(population, boroughs).items()],
        columns=["modzcta", "population", "borough"])
    write_csv(work_dir / AREAS_FILE, areas)

    def top(counter: Counter) -> dict[str, int]:
        return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:TOP_UNMATCHED])

    stats = EnrichStats(
        rows_before_join=rows_before, rows_after_join=rows_after, alias_applied=alias_applied,
        population_status=dict(sorted(counters["population_status"].items())),
        borough_source=dict(sorted(counters["borough_source"].items())),
        borough_check=dict(sorted(counters["borough_check"].items())),
        unmatched_zips=top(unmatched), inconsistent_pairs=top(inconsistent),
        reference={
            "population_areas": len(population.modzcta_population),
            "population_total": sum(v or 0 for v in population.modzcta_population.values()),
            "areas_without_population": population.zero_population_areas,
            "zips_in_population_crosswalk": len(population.zip_to_modzcta),
            "zctas_touching_nyc": len(boroughs.zip_boroughs),
            "zctas_in_several_counties": boroughs.split_zips,
            "blank_key_rows_dropped": boroughs.blank_key_rows,
            "files": reference_meta or {},
        })
    (work_dir / STATS_FILE).write_text(json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8", newline="\n")
    log.info("enrich: %d rows before join, %d after; population %s; borough check %s",
             rows_before, rows_after, stats.population_status, stats.borough_check)
    return stats
