"""The ordered stages of the factory and the code that runs a selection of them.

Each stage is a function that receives the run context. Stages are registered here in
the order they must run; the command line can run all of them or only some.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from data_factory.cleaning import clean_dataset
from data_factory.config import Config
from data_factory.contract import SOURCE_COLUMNS, Scope
from data_factory.enrichment import enrich_dataset
from data_factory.http_client import HttpClient
from data_factory.ingest import fetch_snapshot, snapshot_id
from data_factory.manifest import RunRecord, build_manifest, save_manifest
from data_factory.publish import CASES_FILE, METRICS, QUALITY_FILE, publish, summary_file
from data_factory.exceptions import MissingInputError
from data_factory.raw_store import RawPageStore, Segment, SnapshotChain
from data_factory.references import (
    fetch_references,
    load_borough_reference,
    load_population,
    reference_files,
)
from data_factory.sources import SocrataSource
from data_factory.validation import QUARANTINE_FILE, validate_snapshot

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    config: Config
    since_last_run: bool = False   # fetch only records newer than the previous run
    record: RunRecord = field(default_factory=RunRecord)   # filled in during the run


Stage = Callable[[RunContext], None]


def raw_store(config: Config) -> RawPageStore:
    """The raw snapshot folder for the current configuration."""
    name = snapshot_id(config.source.records_url, list(SOURCE_COLUMNS), config.scope)
    return RawPageStore(config.paths.raw / name)


def references(config: Config):
    """Cached reference files (population, zip-to-county) for the current configuration."""
    return reference_files(config.paths.raw / "references", config.references.population_url,
                           config.references.zcta_county_url)


def chain(config: Config) -> SnapshotChain:
    return SnapshotChain(raw_store(config), config.scope.window_start, config.scope.window_end)


def effective_scope(config: Config) -> Scope:
    """The configured scope, with the window extended to the end of the last increment (D31)."""
    return replace(config.scope, window_end=chain(config).window_end())


def effective_as_of(config: Config) -> pd.Timestamp:
    """The as_of cut-off never lies inside the data: it moves with the window end (D31)."""
    return max(config.as_of, chain(config).window_end())


def today_utc() -> datetime:
    """Today's date at midnight, UTC. A function so tests can fix 'today'."""
    return datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)


def increment_cutoff() -> pd.Timestamp:
    """End (exclusive) of a new increment: midnight starting *yesterday* by the UTC date.

    New York is 4-5 hours behind UTC, so this is never later than the start of the current
    New York day: an increment only ever holds complete New York days, and a day is never
    split between two increments.
    """
    return pd.Timestamp(today_utc() - timedelta(days=1))


def _segment_to_fetch(config: Config, since_last_run: bool) -> Segment | None:
    base = chain(config)
    if not since_last_run:
        return base.base
    if not base.base.store.is_complete():
        raise MissingInputError("--since-last-run needs a complete base snapshot: run the fetch stage first")
    pending = base.unfinished_increment()
    if pending is not None:
        log.info("resuming the unfinished increment %s", pending.store.directory.name)
        return pending
    start, end = base.window_end(), increment_cutoff()
    if end <= start:
        log.info("since last run: data is already complete up to %s; nothing new", start)
        return None
    return base.increment(start, end)


def fetch(context: RunContext) -> None:
    config = context.config
    http = HttpClient(timeout_seconds=config.fetch.timeout_seconds,
                      max_retries=config.fetch.max_retries,
                      backoff_seconds=config.fetch.backoff_seconds)
    segment = _segment_to_fetch(config, context.since_last_run)
    try:
        if segment is not None:
            scope = replace(config.scope, window_start=segment.window_start, window_end=segment.window_end)
            source = SocrataSource(http, config.source.records_url, config.source.metadata_url,
                                   scope, list(SOURCE_COLUMNS))
            result = fetch_snapshot(source, segment.store, SOURCE_COLUMNS, config.fetch.page_size,
                                    extra_info={"window_start": segment.window_start.isoformat(),
                                                "window_end": segment.window_end.isoformat()})
            log.info("fetch: %d new pages, %d new records, segment %s", result.pages_written,
                     result.records_written, segment.store.directory.name)
        fetch_references(http, references(config), config.references.population_url,
                         config.references.zcta_county_url)
    finally:
        context.record.http.update(asdict(http.stats))   # counted even if the fetch fails
    log.info("fetch: HTTP calls %d, attempts %d", http.stats.calls, http.stats.attempts)


def work_dir(config: Config) -> Path:
    return config.paths.work / raw_store(config).directory.name


def validate_stage(context: RunContext) -> None:
    config = context.config
    validate_snapshot([segment.store for segment in chain(config).segments()],
                      work_dir=work_dir(config),
                      quarantine_dir=config.paths.quarantine / raw_store(config).directory.name,
                      scope=effective_scope(config),
                      zip_placeholders=config.zip.placeholders,
                      max_ratio=config.max_quarantine_ratio)


def clean_stage(context: RunContext) -> None:
    config = context.config
    clean_dataset(work_dir(config), effective_as_of(config), config.clean)


def enrich_stage(context: RunContext) -> None:
    config = context.config
    files = references(config)
    population = load_population(files.population)
    boroughs = load_borough_reference(files.zcta_county, config.references.county_boroughs)
    enrich_dataset(work_dir(config), population, boroughs,
                   config.zip.aliases, config.clean.chunk_rows, reference_meta=files.meta())


def publish_stage(context: RunContext) -> None:
    config = context.config
    publish(work_dir(config), config.paths.curated, effective_scope(config), config.aggregate)


STAGE_NAMES = ("fetch", "validate", "clean", "enrich", "publish")
STAGES: dict[str, Stage] = {
    "fetch": fetch,
    "validate": validate_stage,
    "clean": clean_stage,
    "enrich": enrich_stage,
    "publish": publish_stage,
}


def run(context: RunContext, stage_names: list[str]) -> None:
    """Run the given stages, always in the factory's order, then write the run manifest."""
    record = context.record
    try:
        for name in stage_names:
            log.info("stage %s: start", name)
            started = time.perf_counter()
            record.stages.append(name)
            STAGES[name](context)
            record.stage_seconds[name] = round(time.perf_counter() - started, 1)
            log.info("stage %s: done", name)
    except Exception as error:
        record_run(context, status="failed", error=f"{type(error).__name__}: {error}")
        raise
    record_run(context, status="succeeded")


def output_files(config: Config) -> dict[str, Path]:
    """Every published file whose hash goes into the manifest, by name relative to data/."""
    curated = config.paths.curated
    names = [CASES_FILE, QUALITY_FILE] + [summary_file(level, metric)
                                          for level in config.aggregate.levels for metric in METRICS]
    files = {f"curated/{name}": curated / name for name in names}
    snapshot = raw_store(config).directory.name
    files[f"quarantine/{QUARANTINE_FILE}"] = config.paths.quarantine / snapshot / QUARANTINE_FILE
    return files


def record_run(context: RunContext, status: str, error: str | None = None) -> None:
    config = context.config
    segments = []
    for segment in chain(config).segments():
        info = segment.store.completion_info() if segment.store.is_complete() else {}
        segments.append({"name": segment.store.directory.name,
                         "window_start": segment.window_start.isoformat(),
                         "window_end_exclusive": segment.window_end.isoformat(),
                         "fetched_at_utc": info.get("fetched_at_utc"),
                         "pages": info.get("pages"), "raw_records": info.get("records")})
    base = segments[0]
    scope = effective_scope(config)
    files = references(config)
    manifest = build_manifest(
        status=status, error=error, record=context.record, config_path=config.path,
        scope={"agencies": list(scope.agencies),
               "complaint_types": list(scope.complaint_types),
               "window_start": scope.window_start.isoformat(),
               "window_end_exclusive": scope.window_end.isoformat(),
               "as_of": effective_as_of(config).isoformat(),
               "since_last_run": context.since_last_run},
        source={"records_url": config.source.records_url,
                "snapshot": "+".join(s["name"] for s in segments),
                "segments": segments,
                "fetched_at_utc": base["fetched_at_utc"], "pages": base["pages"],
                "raw_records": base["raw_records"], "page_size": config.fetch.page_size},
        work_dir=work_dir(config),
        references=files.meta() if files.exist() else {},
        outputs=output_files(config),
    )
    save_manifest(manifest, config.paths.curated, config.paths.work / "manifests")
