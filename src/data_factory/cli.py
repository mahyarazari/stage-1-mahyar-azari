"""Command line entry point: `data-factory` (or `python -m data_factory`).

Examples:
    data-factory run                          # every stage, from scratch to final outputs
    data-factory run --stages fetch           # only one stage
    data-factory run --since-last-run         # only records newer than the previous run
    data-factory check-config                 # validate the configuration and stop
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_factory import pipeline
from data_factory.config import load_config
from data_factory.exceptions import ExitCode, FactoryError

log = logging.getLogger("data_factory")

DEFAULT_CONFIG = Path("factory.toml")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def _stage_list(text: str) -> list[str]:
    """Parse 'fetch,clean' into stage names, in the factory's order whatever the input order."""
    requested = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in requested if name not in pipeline.STAGES]
    if not requested or unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stage(s) {unknown}; choose from {list(pipeline.STAGE_NAMES)}")
    return [name for name in pipeline.STAGE_NAMES if name in requested]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="data-factory",
                                     description="Build a clean NYC 311 dataset from the live API.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"configuration file (default: {DEFAULT_CONFIG})")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run the factory (all stages by default)")
    run.add_argument("--stages", type=_stage_list, default=list(pipeline.STAGE_NAMES),
                     help=f"comma-separated subset of: {','.join(pipeline.STAGE_NAMES)}")
    run.add_argument("--since-last-run", action="store_true",
                     help="fetch only records created after the previous run")

    commands.add_parser("check-config", help="load and validate the configuration, then stop")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=LOG_FORMAT)
    try:
        config = load_config(args.config)
        if args.command == "check-config":
            log.info("configuration is valid: %s", config.path)
            return ExitCode.OK
        pipeline.run(pipeline.RunContext(config=config, since_last_run=args.since_last_run),
                     args.stages)
        return ExitCode.OK
    except FactoryError as error:
        log.error("%s: %s", type(error).__name__, error)
        return error.exit_code
    except Exception:  # a bug: show the full traceback
        log.exception("unexpected error")
        return ExitCode.UNEXPECTED
