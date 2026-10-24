"""Project exception hierarchy and the process exit code of each error family.

A caller (a shell script, a scheduler, the evaluator) can tell from the exit code alone
whether a run failed because of the network, the data, or the configuration.
"""
from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    UNEXPECTED = 1            # a bug: anything that is not a FactoryError
    # 2 is left to argparse, which uses it for command-line usage errors.
    CONFIG = 10
    NETWORK = 20
    INVALID_RESPONSE = 21
    DATA_VALIDATION = 30
    QUARANTINE_THRESHOLD = 31
    SOURCE_SCHEMA_CHANGED = 32
    MISSING_INPUT = 40


class FactoryError(Exception):
    """Base class of every error the factory raises on purpose."""

    exit_code = ExitCode.UNEXPECTED


class ConfigError(FactoryError):
    """The configuration file is missing, unreadable or contains an invalid value."""

    exit_code = ExitCode.CONFIG


class NetworkError(FactoryError):
    """The source could not be reached, or kept failing after every retry."""

    exit_code = ExitCode.NETWORK


class InvalidResponseError(NetworkError):
    """The source answered, but not with what was asked for (e.g. HTTP 200 with an HTML page)."""

    exit_code = ExitCode.INVALID_RESPONSE


class DataValidationError(FactoryError):
    """The data itself cannot be trusted enough to continue."""

    exit_code = ExitCode.DATA_VALIDATION


class QuarantineThresholdError(DataValidationError):
    """Too large a share of records was quarantined: the run stops instead of publishing."""

    exit_code = ExitCode.QUARANTINE_THRESHOLD


class SourceSchemaChangedError(DataValidationError):
    """A column the contract needs was removed from the source or changed type."""

    exit_code = ExitCode.SOURCE_SCHEMA_CHANGED


class HttpStatusError(NetworkError):
    """The source answered with an error status that retrying will not fix (e.g. 400, 404)."""

    def __init__(self, message: str, status: int, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class MissingInputError(FactoryError):
    """A stage was asked to run before the stage that produces its input (e.g. validate before fetch)."""

    exit_code = ExitCode.MISSING_INPUT
