"""Load and validate the factory configuration (a TOML file).

Every tunable value of the factory lives in the configuration file, never in the code.
Relative paths are resolved against the folder of the configuration file, so the factory
behaves the same whatever folder it is started from.

TOML is read with `tomllib` from the standard library (Python 3.11+), so no extra
dependency is needed. Unlike JSON, TOML allows comments, which keeps each setting's
reason next to the setting.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data_factory.contract import BOROUGHS, Scope
from data_factory.exceptions import ConfigError

ZIP_PATTERN = re.compile(r"\d{5}")
AGGREGATE_LEVELS = ("borough", "modzcta")


@dataclass(frozen=True)
class SourceConfig:
    records_url: str
    metadata_url: str


@dataclass(frozen=True)
class FetchConfig:
    page_size: int
    max_retries: int
    backoff_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    curated: Path
    quarantine: Path
    work: Path        # intermediate results passed from one stage to the next


@dataclass(frozen=True)
class ReferenceConfig:
    population_url: str
    zcta_county_url: str
    county_boroughs: dict[str, str]   # county FIPS code -> borough name as written in 311


@dataclass(frozen=True)
class ZipConfig:
    placeholders: tuple[str, ...]
    aliases: dict[str, str]


@dataclass(frozen=True)
class AggregateConfig:
    levels: tuple[str, ...]
    min_cases_per_cell: int
    rate_per: int            # rates are complaints per this many residents


@dataclass(frozen=True)
class CleanConfig:
    chunk_rows: int                       # rows processed at a time
    unspecified_values: tuple[str, ...]   # source text meaning "unknown" (becomes missing)
    latitude_range: tuple[float, float]   # coordinates outside the city become missing
    longitude_range: tuple[float, float]


@dataclass(frozen=True)
class Config:
    path: Path
    source: SourceConfig
    scope: Scope
    as_of: pd.Timestamp   # cut-off: a case closed after it counts as still open (D24)
    fetch: FetchConfig
    paths: PathsConfig
    references: ReferenceConfig
    zip: ZipConfig
    max_quarantine_ratio: float
    clean: CleanConfig
    aggregate: AggregateConfig


# ---------------------------------------------------------------- small readers
def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing section [{name}]")
    return value


def _get(section: dict[str, Any], section_name: str, key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in section:
        raise ConfigError(f"missing key '{key}' in [{section_name}]")
    value = section[key]
    # bool is a subclass of int: refuse `true` where a number is expected.
    if not isinstance(value, kind) or (isinstance(value, bool) and bool not in _as_tuple(kind)):
        raise ConfigError(f"[{section_name}] {key} must be {_kind_name(kind)}, got {value!r}")
    return value


def _as_tuple(kind: type | tuple[type, ...]) -> tuple[type, ...]:
    return kind if isinstance(kind, tuple) else (kind,)


def _kind_name(kind: type | tuple[type, ...]) -> str:
    return " or ".join(k.__name__ for k in _as_tuple(kind))


def _positive(value: float, where: str) -> float:
    if value <= 0:
        raise ConfigError(f"{where} must be greater than 0, got {value}")
    return value


def _text_list(section: dict[str, Any], section_name: str, key: str) -> tuple[str, ...]:
    values = _get(section, section_name, key, list)
    if not values or not all(isinstance(v, str) and v.strip() for v in values):
        raise ConfigError(f"[{section_name}] {key} must be a non-empty list of text values")
    return tuple(values)


# ---------------------------------------------------------------- sections
def _read_scope(data: dict[str, Any]) -> Scope:
    section = _section(data, "scope")
    start = _get(section, "scope", "window_start", datetime)
    end = _get(section, "scope", "window_end", datetime)
    if start.tzinfo or end.tzinfo:
        raise ConfigError("[scope] window dates must not carry a time zone (source dates are local NYC time)")
    if start >= end:
        raise ConfigError(f"[scope] window_start ({start}) must be before window_end ({end})")
    return Scope(
        agencies=_text_list(section, "scope", "agencies"),
        complaint_types=_text_list(section, "scope", "complaint_types"),
        window_start=pd.Timestamp(start),
        window_end=pd.Timestamp(end),
    )


def _read_as_of(data: dict[str, Any], scope: Scope) -> pd.Timestamp:
    section = _section(data, "scope")
    as_of = _get(section, "scope", "as_of", datetime)
    if as_of.tzinfo:
        raise ConfigError("[scope] as_of must not carry a time zone")
    if pd.Timestamp(as_of) < scope.window_end:
        raise ConfigError(f"[scope] as_of ({as_of}) must not be before window_end ({scope.window_end})")
    return pd.Timestamp(as_of)


def _read_fetch(data: dict[str, Any]) -> FetchConfig:
    section = _section(data, "fetch")
    max_retries = _get(section, "fetch", "max_retries", int)
    if max_retries < 0:
        raise ConfigError(f"[fetch] max_retries must be 0 or more, got {max_retries}")
    return FetchConfig(
        page_size=int(_positive(_get(section, "fetch", "page_size", int), "[fetch] page_size")),
        max_retries=max_retries,
        backoff_seconds=float(_positive(_get(section, "fetch", "backoff_seconds", (int, float)),
                                        "[fetch] backoff_seconds")),
        timeout_seconds=float(_positive(_get(section, "fetch", "timeout_seconds", (int, float)),
                                        "[fetch] timeout_seconds")),
    )


def _read_paths(data: dict[str, Any], base: Path) -> PathsConfig:
    section = _section(data, "paths")

    def resolve(key: str) -> Path:
        path = Path(_get(section, "paths", key, str))
        return path if path.is_absolute() else (base / path).resolve()

    return PathsConfig(raw=resolve("raw"), curated=resolve("curated"),
                       quarantine=resolve("quarantine"), work=resolve("work"))


def _read_references(data: dict[str, Any]) -> ReferenceConfig:
    section = _section(data, "references")
    counties = _get(section, "references", "county_boroughs", dict)
    for code, borough in counties.items():
        if not re.fullmatch(r"\d{5}", code) or borough not in BOROUGHS:
            raise ConfigError(f"[references.county_boroughs] '{code}' = {borough!r}: "
                              f"need a 5-digit county code and one of {list(BOROUGHS)}")
    if sorted(counties.values()) != sorted(BOROUGHS):
        raise ConfigError("[references.county_boroughs] must map exactly one county to each borough")
    return ReferenceConfig(
        population_url=_get(section, "references", "population_url", str),
        zcta_county_url=_get(section, "references", "zcta_county_url", str),
        county_boroughs=dict(counties),
    )


def _read_zip(data: dict[str, Any]) -> ZipConfig:
    section = _section(data, "zip")
    placeholders = _get(section, "zip", "placeholders", list)
    if not all(isinstance(p, str) for p in placeholders):
        raise ConfigError("[zip] placeholders must be a list of text values")
    aliases = _get(section, "zip", "aliases", dict)
    for old, new in aliases.items():
        if not (isinstance(new, str) and ZIP_PATTERN.fullmatch(old) and ZIP_PATTERN.fullmatch(new)):
            raise ConfigError(f"[zip.aliases] '{old}' = {new!r}: both sides must be 5-digit zips")
    return ZipConfig(placeholders=tuple(placeholders), aliases=dict(aliases))


def _read_quarantine(data: dict[str, Any]) -> float:
    section = _section(data, "quarantine")
    ratio = float(_get(section, "quarantine", "max_ratio", (int, float)))
    if not 0 < ratio <= 1:
        raise ConfigError(f"[quarantine] max_ratio must be in (0, 1], got {ratio}")
    return ratio


def _range(section: dict[str, Any], key: str) -> tuple[float, float]:
    values = _get(section, "clean", key, list)
    if (len(values) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
            or values[0] >= values[1]):
        raise ConfigError(f"[clean] {key} must be [low, high] with low < high, got {values!r}")
    return float(values[0]), float(values[1])


def _read_clean(data: dict[str, Any]) -> CleanConfig:
    section = _section(data, "clean")
    return CleanConfig(
        chunk_rows=int(_positive(_get(section, "clean", "chunk_rows", int), "[clean] chunk_rows")),
        unspecified_values=_text_list(section, "clean", "unspecified_values"),
        latitude_range=_range(section, "latitude_range"),
        longitude_range=_range(section, "longitude_range"),
    )


def _read_aggregate(data: dict[str, Any]) -> AggregateConfig:
    section = _section(data, "aggregate")
    levels = _text_list(section, "aggregate", "levels")
    unknown = sorted(set(levels) - set(AGGREGATE_LEVELS))
    if unknown:
        raise ConfigError(f"[aggregate] unknown levels {unknown}; allowed: {list(AGGREGATE_LEVELS)}")
    min_cases = _get(section, "aggregate", "min_cases_per_cell", int)
    rate_per = _get(section, "aggregate", "rate_per", int)
    return AggregateConfig(levels=levels,
                           min_cases_per_cell=int(_positive(min_cases, "[aggregate] min_cases_per_cell")),
                           rate_per=int(_positive(rate_per, "[aggregate] rate_per")))


# ---------------------------------------------------------------- entry point
def load_config(path: str | Path) -> Config:
    """Read, validate and return the configuration. Any problem raises ConfigError."""
    path = Path(path).resolve()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"configuration file not found: {path}") from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"configuration file is not valid TOML ({path}): {error}") from None

    source = _section(data, "source")
    scope = _read_scope(data)
    return Config(
        path=path,
        source=SourceConfig(
            records_url=_get(source, "source", "records_url", str),
            metadata_url=_get(source, "source", "metadata_url", str),
        ),
        scope=scope,
        as_of=_read_as_of(data, scope),
        fetch=_read_fetch(data),
        paths=_read_paths(data, base=path.parent),
        references=_read_references(data),
        zip=_read_zip(data),
        max_quarantine_ratio=_read_quarantine(data),
        clean=_read_clean(data),
        aggregate=_read_aggregate(data),
    )
