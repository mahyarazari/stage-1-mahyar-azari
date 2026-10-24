"""The ordered stages of the factory and the code that runs a selection of them.

Each stage is a function that receives the run context. Stages are registered here in
the order they must run; the command line can run all of them or only some.
Stages are filled in one by one as they are built; until then they say so clearly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from data_factory.config import Config
from data_factory.contract import SOURCE_COLUMNS
from data_factory.http_client import HttpClient
from data_factory.ingest import fetch_snapshot, snapshot_id
from data_factory.raw_store import RawPageStore
from data_factory.sources import SocrataSource

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunContext:
    config: Config
    since_last_run: bool = False   # fetch only records newer than the previous run


Stage = Callable[[RunContext], None]


def _not_built_yet(name: str) -> Stage:
    def stage(context: RunContext) -> None:
        raise NotImplementedError(f"stage '{name}' is not built yet")
    return stage


def raw_store(config: Config) -> RawPageStore:
    """The raw snapshot folder for the current configuration."""
    name = snapshot_id(config.source.records_url, list(SOURCE_COLUMNS), config.scope)
    return RawPageStore(config.paths.raw / name)


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
    result = fetch_snapshot(source, store, SOURCE_COLUMNS, config.fetch.page_size)
    log.info("fetch: %d new pages, %d new records, snapshot %s; HTTP calls %d, attempts %d",
             result.pages_written, result.records_written, store.directory.name,
             http.stats.calls, http.stats.attempts)


STAGE_NAMES = ("fetch", "validate", "clean", "enrich", "publish")
STAGES: dict[str, Stage] = {name: _not_built_yet(name) for name in STAGE_NAMES}
STAGES["fetch"] = fetch


def run(context: RunContext, stage_names: list[str]) -> None:
    """Run the given stages, always in the factory's order."""
    for name in stage_names:
        log.info("stage %s: start", name)
        STAGES[name](context)
        log.info("stage %s: done", name)
