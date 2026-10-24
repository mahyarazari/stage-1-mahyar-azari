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


STAGE_NAMES = ("fetch", "validate", "clean", "enrich", "publish")
STAGES: dict[str, Stage] = {name: _not_built_yet(name) for name in STAGE_NAMES}


def run(context: RunContext, stage_names: list[str]) -> None:
    """Run the given stages, always in the factory's order."""
    for name in stage_names:
        log.info("stage %s: start", name)
        STAGES[name](context)
        log.info("stage %s: done", name)
