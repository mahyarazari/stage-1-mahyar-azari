"""The ordered stages of the factory and the code that runs a selection of them.

Each stage is a function that receives the run context. Stages are registered here in
the order they must run; the command line can run all of them or only some.
"""
from __future__ import annotations

from pathlib import Path

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from data_factory.cleaning import clean_dataset
from data_factory.config import Config
from data_factory.contract import SOURCE_COLUMNS
from data_factory.enrichment import enrich_dataset
from data_factory.http_client import HttpClient
from data_factory.ingest import fetch_snapshot, snapshot_id
from data_factory.manifest import RunRecord, build_manifest, save_manifest
from data_factory.publish import CASES_FILE, QUALITY_FILE, publish, summary_file
from data_factory.publish import METRICS
from data_factory.raw_store import RawPageStore
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


def fetch(context: RunContext) -> None:
    if context.since_last_run:
        raise NotImplementedError("--since-last-run is not built yet")
    config = context.config
    http = HttpClient(timeout_seconds=config.fetch.timeout_seconds,
                      max_retries=config.fetch.max_retries,
                      backoff_seconds=config.fetch.backoff_seconds)
    source = SocrataSource(http, config.source.records_url, config.source.metadata_url,
                           config.scope, list(SOURCE_COLUMNS))
    store = raw_store(config)
    try:
        result = fetch_snapshot(source, store, SOURCE_COLUMNS, config.fetch.page_size)
        fetch_references(http, references(config), config.references.population_url,
                         config.references.zcta_county_url)
    finally:
        context.record.http.update(asdict(http.stats))   # counted even if the fetch fails
    log.info("fetch: %d new pages, %d new records, snapshot %s; HTTP calls %d, attempts %d",
             result.pages_written, result.records_written, store.directory.name,
             http.stats.calls, http.stats.attempts)


def validate_stage(context: RunContext) -> None:
    config = context.config
    store = raw_store(config)
    validate_snapshot(store,
                      work_dir=config.paths.work / store.directory.name,
                      quarantine_dir=config.paths.quarantine / store.directory.name,
                      scope=config.scope,
                      zip_placeholders=config.zip.placeholders,
                      max_ratio=config.max_quarantine_ratio)


def clean_stage(context: RunContext) -> None:
    config = context.config
    store = raw_store(config)
    clean_dataset(config.paths.work / store.directory.name, config.as_of, config.clean)


def enrich_stage(context: RunContext) -> None:
    config = context.config
    files = references(config)
    population = load_population(files.population)
    boroughs = load_borough_reference(files.zcta_county, config.references.county_boroughs)
    enrich_dataset(config.paths.work / raw_store(config).directory.name, population, boroughs,
                   config.zip.aliases, config.clean.chunk_rows, reference_meta=files.meta())


def publish_stage(context: RunContext) -> None:
    config = context.config
    publish(config.paths.work / raw_store(config).directory.name, config.paths.curated,
            config.scope, config.aggregate)


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
    store = raw_store(config)
    info = store.completion_info() if store.is_complete() else {}
    files = references(config)
    manifest = build_manifest(
        status=status, error=error, record=context.record, config_path=config.path,
        scope={"agencies": list(config.scope.agencies),
               "complaint_types": list(config.scope.complaint_types),
               "window_start": config.scope.window_start.isoformat(),
               "window_end_exclusive": config.scope.window_end.isoformat(),
               "as_of": config.as_of.isoformat(),
               "since_last_run": context.since_last_run},
        source={"records_url": config.source.records_url, "snapshot": store.directory.name,
                "fetched_at_utc": info.get("fetched_at_utc"), "pages": info.get("pages"),
                "raw_records": info.get("records"), "page_size": config.fetch.page_size},
        work_dir=config.paths.work / store.directory.name,
        references=files.meta() if files.exist() else {},
        outputs=output_files(config),
    )
    save_manifest(manifest, config.paths.curated, config.paths.work / "manifests")
