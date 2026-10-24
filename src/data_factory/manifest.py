"""Run manifest: what a run did, with what code and inputs, and a hash of every output.

Written to data/curated/run_manifest.json after every successful run, and a copy of every
run (successful or failed) to data/work/manifests/. The manifest itself contains times, so
it is never one of the hashed outputs.

Determinism check (D30): before overwriting, the previous manifest is read. If the previous
run had the same inputs (snapshot, configuration file, code commit) its output hashes must
be identical; the comparison and any changed file are written into the new manifest, so
"two runs give the same hashes" is recorded evidence, not a claim.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data_factory import __version__

log = logging.getLogger(__name__)

MANIFEST_FILE = "run_manifest.json"
HISTORY_DIR = "manifests"
HASH_CHUNK_BYTES = 1 << 20
GIT_TIMEOUT_SECONDS = 10
STATS_FILES = {
    "validate": "validate_stats.json",
    "clean": "clean_stats.json",
    "enrich": "enrich_stats.json",
    "publish": "publish_stats.json",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


@dataclass
class RunRecord:
    """Filled in while the run goes on; turned into the manifest at the end."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started_at: datetime = field(default_factory=utc_now)
    started_clock: float = field(default_factory=time.perf_counter)
    stages: list[str] = field(default_factory=list)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    http: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- pieces
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)      # minus the header line


def describe_outputs(files: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """name -> sha256, size and rows of every output that exists, sorted by name."""
    return {name: {"sha256": file_sha256(path), "bytes": path.stat().st_size, "rows": _rows(path)}
            for name, path in sorted(files.items()) if path.exists()}


def code_version(folder: Path) -> dict[str, Any]:
    """Package version and, when the project is a Git checkout, the commit and dirty state."""
    version: dict[str, Any] = {"package": __version__, "git_commit": "unknown", "git_dirty": None}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=folder, capture_output=True,
                                text=True, timeout=GIT_TIMEOUT_SECONDS, check=True).stdout.strip()
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                 cwd=folder, capture_output=True, text=True,
                                 timeout=GIT_TIMEOUT_SECONDS, check=True).stdout.strip()
        version.update(git_commit=commit, git_dirty=bool(changes))
    except (OSError, subprocess.SubprocessError):
        pass
    return version


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def compare_with_previous(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None or previous.get("status") != "succeeded":
        return {"previous_run": None, "same_inputs": None, "identical_outputs": None, "changed": []}
    same_inputs = all(previous.get("inputs", {}).get(key) == current["inputs"].get(key)
                      for key in ("snapshot", "config_sha256", "git_commit"))
    old, new = previous.get("outputs", {}), current["outputs"]
    changed = sorted(name for name in set(old) | set(new)
                     if old.get(name, {}).get("sha256") != new.get(name, {}).get("sha256"))
    return {"previous_run": previous.get("run", {}).get("started_at_utc"),
            "same_inputs": same_inputs, "identical_outputs": not changed, "changed": changed}


# ---------------------------------------------------------------- manifest
def build_manifest(*, status: str, record: RunRecord, config_path: Path, scope: dict[str, Any],
                   source: dict[str, Any], work_dir: Path, references: dict[str, Any],
                   outputs: dict[str, Path], error: str | None = None) -> dict[str, Any]:
    finished = utc_now()
    stats = {stage: _read_json(work_dir / name) for stage, name in STATS_FILES.items()}
    validate = stats["validate"] or {}
    code = code_version(config_path.parent)
    manifest = {
        "status": status,
        "error": error,
        "run": {
            "run_id": record.run_id,
            "started_at_utc": record.started_at.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - record.started_clock, 1),
            "stages": record.stages,
            "stage_seconds": record.stage_seconds,
        },
        "code": code,
        "inputs": {
            "snapshot": source.get("snapshot"),
            "config_file": config_path.name,
            "config_sha256": file_sha256(config_path),
            "git_commit": code["git_commit"],
        },
        "scope": scope,
        "source": source,
        "references": references,
        "requests": {"this_run": record.http, "full_download": _full_download_requests(source)},
        "counts": {
            "raw_records": validate.get("raw_records"),
            "passed_contract": validate.get("valid_records"),
            "quarantined": validate.get("quarantined_records"),
            "quarantine_ratio": validate.get("quarantine_ratio"),
            "quarantine_by_reason": validate.get("reasons"),
            "clean": stats["clean"],
            "enrich": stats["enrich"],
            "publish": stats["publish"],
        },
        "outputs": describe_outputs(outputs),
    }
    return manifest


def _full_download_requests(source: dict[str, Any]) -> dict[str, Any] | None:
    pages = source.get("pages")
    if pages is None:
        return None
    # 1 schema check + one request per page (+1 empty answer when the last page was full)
    # + 2 references. See D22 for why this is the minimum.
    full_last_page = source.get("raw_records", 0) == pages * source.get("page_size", 0)
    return {"schema_check": 1, "pages": pages + int(full_last_page), "references": 2,
            "total": 1 + pages + int(full_last_page) + 2}


def save_manifest(manifest: dict[str, Any], curated_dir: Path, history_dir: Path) -> dict[str, Any]:
    """Compare with the previous successful run, then write the manifest and its history copy."""
    if manifest["status"] == "succeeded":
        previous = _read_json(curated_dir / MANIFEST_FILE)
        manifest["determinism"] = compare_with_previous(previous, manifest)
        check = manifest["determinism"]
        if check["same_inputs"] and not check["identical_outputs"]:
            log.warning("same inputs as the previous run but outputs differ: %s", check["changed"])
        elif check["same_inputs"]:
            log.info("determinism: same inputs as the previous run, all %d output hashes identical",
                     len(manifest["outputs"]))
        _write_json(curated_dir / MANIFEST_FILE, manifest)
    stamp = manifest["run"]["started_at_utc"].replace(":", "")
    name = f"{stamp}_{manifest['run']['run_id']}_{manifest['status']}.json"
    _write_json(history_dir / name, manifest)
    return manifest
