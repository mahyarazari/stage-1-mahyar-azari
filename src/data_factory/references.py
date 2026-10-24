"""Reference tables: population per zip (MODZCTA) and zip -> borough (Census ZCTA-county file).

Download: each reference is fetched once and cached in data/raw/references/ with a small
meta file (URL, retrieval time). Like the main snapshot it is never downloaded again, so
later runs are offline and deterministic (D24). The retrieval time is kept for the report.

Validation (lesson of the NaN-matches-NaN bug, D7): a reference with blank keys, unknown
columns, a zip mapped to two areas, or negative population is refused with
ReferenceDataError instead of being joined.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_factory.exceptions import InvalidResponseError, MissingInputError, ReferenceDataError
from data_factory.http_client import HttpClient

log = logging.getLogger(__name__)

ZIP5 = re.compile(r"\d{5}")
POPULATION_SELECT = "modzcta, label, zcta, pop_est"
POPULATION_LIMIT = 1000          # far above the ~178 areas; a full page means truncation
OUTSIDE_NYC = "OUTSIDE_NYC"
ENCODING = "utf-8"


# ---------------------------------------------------------------- cache
def _cache_name(stem: str, url: str, suffix: str) -> str:
    return f"{stem}-{hashlib.sha256(url.encode()).hexdigest()[:8]}{suffix}"


def _write_once(path: Path, text: str, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding=ENCODING, newline="") as handle:   # keep bytes as sent
        handle.write(text)
    os.replace(temporary, path)
    meta = {"url": url, "retrieved_at_utc": datetime.now(timezone.utc).replace(
        tzinfo=None, microsecond=0).isoformat()}
    path.with_name(path.name + ".meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding=ENCODING, newline="\n")


@dataclass(frozen=True)
class ReferenceFiles:
    population: Path
    zcta_county: Path

    def exist(self) -> bool:
        return self.population.exists() and self.zcta_county.exists()

    def meta(self) -> dict[str, Any]:
        return {name: json.loads(path.with_name(path.name + ".meta.json").read_text(ENCODING))
                for name, path in (("population", self.population), ("zcta_county", self.zcta_county))}


def reference_files(cache_dir: Path, population_url: str, zcta_county_url: str) -> ReferenceFiles:
    return ReferenceFiles(population=cache_dir / _cache_name("modzcta", population_url, ".json"),
                          zcta_county=cache_dir / _cache_name("zcta_county", zcta_county_url, ".txt"))


def fetch_references(http: HttpClient, files: ReferenceFiles, population_url: str,
                     zcta_county_url: str) -> None:
    """Download each reference that is not cached yet."""
    if not files.population.exists():
        rows = http.get_json(population_url, {"$select": POPULATION_SELECT,
                                               "$order": "modzcta", "$limit": POPULATION_LIMIT})
        if not isinstance(rows, list):
            raise InvalidResponseError(f"{population_url}: expected a JSON list")
        if len(rows) >= POPULATION_LIMIT:
            raise InvalidResponseError(f"{population_url}: {len(rows)} rows, answer may be truncated")
        _write_once(files.population, json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=1), population_url)
        log.info("reference cached: %s (%d rows)", files.population.name, len(rows))
    if not files.zcta_county.exists():
        _write_once(files.zcta_county, http.get_text(zcta_county_url), zcta_county_url)
        log.info("reference cached: %s", files.zcta_county.name)


# ---------------------------------------------------------------- population
@dataclass(frozen=True)
class PopulationReference:
    zip_to_modzcta: dict[str, str]
    modzcta_population: dict[str, float | None]   # None: no population given

    @property
    def zero_population_areas(self) -> list[str]:
        return sorted(k for k, v in self.modzcta_population.items() if not v)


def parse_population(rows: Any) -> PopulationReference:
    if not isinstance(rows, list) or not rows:
        raise ReferenceDataError("population reference is empty or not a list")
    zip_to_modzcta: dict[str, str] = {}
    population: dict[str, float | None] = {}
    for row in rows:
        area = str(row.get("modzcta", "")).strip()
        if not ZIP5.fullmatch(area):
            raise ReferenceDataError(f"population row with an invalid area key {area!r}")
        if area in population:
            raise ReferenceDataError(f"area {area} appears twice in the population reference")
        value = row.get("pop_est")
        try:
            number = None if value in (None, "") else float(value)
        except (TypeError, ValueError):
            raise ReferenceDataError(f"area {area}: population {value!r} is not a number") from None
        if number is not None and number < 0:
            raise ReferenceDataError(f"area {area}: negative population {number}")
        population[area] = number
        for zip_code in str(row.get("zcta", "")).split(","):
            zip_code = zip_code.strip()
            if not zip_code:
                continue
            if not ZIP5.fullmatch(zip_code):
                raise ReferenceDataError(f"area {area}: invalid zip {zip_code!r} in its zip list")
            if zip_to_modzcta.get(zip_code, area) != area:   # would duplicate rows in a join
                raise ReferenceDataError(f"zip {zip_code} belongs to two areas: "
                                         f"{zip_to_modzcta[zip_code]} and {area}")
            zip_to_modzcta[zip_code] = area
    return PopulationReference(zip_to_modzcta, population)


def load_population(path: Path) -> PopulationReference:
    if not path.exists():
        raise MissingInputError(f"{path} not found: run the fetch stage first")
    return parse_population(json.loads(path.read_text(ENCODING)))


# ---------------------------------------------------------------- zip -> borough
@dataclass(frozen=True)
class BoroughReference:
    zip_boroughs: dict[str, tuple[str, ...]]      # NYC boroughs a ZCTA touches (sorted)
    zip_land: dict[str, dict[str, float]]          # land area per borough, OUTSIDE_NYC included
    blank_key_rows: int = 0
    split_zips: list[str] = field(default_factory=list)


def _column(columns, *parts: str) -> str:
    matches = [c for c in columns if all(p in c.upper() for p in parts)]
    if len(matches) != 1:
        raise ReferenceDataError(f"zip-county reference: expected one column with {parts}, found {matches}")
    return matches[0]


def parse_zcta_county(text: str, county_boroughs: dict[str, str]) -> BoroughReference:
    table = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    zcta = table[_column(table.columns, "GEOID", "ZCTA")].str.strip()
    county = table[_column(table.columns, "GEOID", "COUNTY")].str.strip()
    land = pd.to_numeric(table[_column(table.columns, "AREALAND_PART")], errors="coerce")
    if land.isna().any():
        raise ReferenceDataError("zip-county reference: AREALAND_PART has non-numeric values")

    blank = zcta == ""                    # county land outside any ZCTA: never a join key
    pieces = pd.DataFrame({"zcta": zcta, "county": county, "land": land})[~blank]
    nyc_zctas = set(pieces.loc[pieces["county"].isin(county_boroughs), "zcta"])
    if not nyc_zctas:
        raise ReferenceDataError("zip-county reference has no ZCTA in the configured counties")
    pieces = pieces[pieces["zcta"].isin(nyc_zctas)]
    pieces = pieces.assign(borough=pieces["county"].map(county_boroughs).fillna(OUTSIDE_NYC))

    zip_land: dict[str, dict[str, float]] = {}
    for (code, borough), area in pieces.groupby(["zcta", "borough"])["land"].sum().items():
        zip_land.setdefault(code, {})[borough] = float(area)
    zip_boroughs = {code: tuple(sorted(b for b in areas if b != OUTSIDE_NYC))
                    for code, areas in zip_land.items()}
    split = sorted(code for code, areas in zip_land.items() if len(areas) > 1)
    return BoroughReference(zip_boroughs, zip_land, int(blank.sum()), split)


def load_borough_reference(path: Path, county_boroughs: dict[str, str]) -> BoroughReference:
    if not path.exists():
        raise MissingInputError(f"{path} not found: run the fetch stage first")
    with path.open(encoding=ENCODING, newline="") as handle:
        return parse_zcta_county(handle.read(), county_boroughs)


# ---------------------------------------------------------------- area -> borough
def area_boroughs(population: PopulationReference, boroughs: BoroughReference) -> dict[str, str | None]:
    """Borough of each population area: the borough holding most of its zips' land (D7 limit).

    Ties are broken alphabetically so the result is deterministic. An area none of whose
    zips is in the Census file gets None.
    """
    land: dict[str, dict[str, float]] = {}
    for zip_code, area in population.zip_to_modzcta.items():
        for borough, value in boroughs.zip_land.get(zip_code, {}).items():
            if borough != OUTSIDE_NYC:
                land.setdefault(area, {}).setdefault(borough, 0.0)
                land[area][borough] += value
    result: dict[str, str | None] = {}
    for area in sorted(population.modzcta_population):
        areas = land.get(area)
        result[area] = None if not areas else sorted(areas.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return result
