"""
src/utils/io.py
===============
Atomic file I/O helpers with Pandera schema validation.

DESIGN PRINCIPLE: Every file written by any module is validated against its
declared schema before being persisted. A schema violation aborts the write
with a descriptive error — never silently truncates or coerces data.
(SRS FR-0.2, NFR-R2)

Atomic write protocol (NFR-R2): write to a temp file in the same directory,
then os.replace() — which is atomic on POSIX and Windows NTFS. This means a
crash mid-write leaves either the fully-written new file or the original file
intact, never a partially-written or corrupted file.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pandera as pa

from src.utils.schemas import SCHEMA_REGISTRY, validate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ===========================================================================
# Parquet I/O
# ===========================================================================


def write_parquet(
    df: pd.DataFrame,
    path: Path | str,
    schema_name: str,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Validate df against schema_name, then atomically write to a Parquet file.

    Args:
        df: DataFrame to write.
        path: Destination file path. Parent directories are created if needed.
        schema_name: Schema key from SCHEMA_REGISTRY (e.g. "trade_log").
        overwrite: If False and path exists, raises FileExistsError.
                   The run_id duplicate-protection check uses overwrite=False.

    Returns:
        Resolved absolute path of the written file.

    Raises:
        pandera.errors.SchemaErrors: On validation failure (lazy=True collects all errors).
        FileExistsError: If overwrite=False and file already exists.
        KeyError: If schema_name is unknown.
    """
    path = Path(path).resolve()

    if not overwrite and path.exists():
        raise FileExistsError(
            f"File already exists at {path}. "
            "Use overwrite=True or choose a new run_id."
        )

    # Schema validation — raises pa.errors.SchemaErrors on failure
    try:
        validated_df = validate(df, schema_name)
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Schema validation failed for '%s' before write to %s:\n%s",
            schema_name,
            path,
            exc.failure_cases.to_string() if hasattr(exc, "failure_cases") else str(exc),
        )
        raise

    # Create parent directories
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: temp file in same directory → os.replace()
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent, suffix=".parquet.tmp"
    )
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        validated_df.to_parquet(tmp_path, index=False, engine="pyarrow")
        os.replace(tmp_path, path)  # atomic on POSIX + Windows NTFS
        logger.debug("Wrote %d rows to %s (schema=%s)", len(validated_df), path, schema_name)
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return path


def read_parquet(
    path: Path | str,
    schema_name: str | None = None,
) -> pd.DataFrame:
    """
    Read a Parquet file and optionally validate against a schema.

    Args:
        path: Path to the Parquet file.
        schema_name: If provided, validates the loaded DataFrame. Useful for
                     catching schema drift when reading files written by other modules.

    Returns:
        Loaded (and optionally validated) DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pandera.errors.SchemaErrors: If schema_name is provided and validation fails.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")

    if schema_name is not None:
        df = validate(df, schema_name)

    logger.debug("Read %d rows from %s", len(df), path)
    return df


# ===========================================================================
# CSV I/O (for human-readable outputs and legacy compatibility)
# ===========================================================================


def write_csv(
    df: pd.DataFrame,
    path: Path | str,
    schema_name: str,
    *,
    overwrite: bool = True,
) -> Path:
    """
    Validate df against schema_name, then atomically write to CSV.

    See write_parquet() for full argument documentation — identical contract.
    """
    path = Path(path).resolve()

    if not overwrite and path.exists():
        raise FileExistsError(f"File already exists at {path}.")

    try:
        validated_df = validate(df, schema_name)
    except pa.errors.SchemaErrors as exc:
        logger.error("Schema validation failed for '%s' before CSV write to %s", schema_name, path)
        raise

    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        validated_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        logger.debug("Wrote %d rows to %s (schema=%s)", len(validated_df), path, schema_name)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    return path


def read_csv(
    path: Path | str,
    schema_name: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file and optionally validate against a schema."""
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if schema_name is not None:
        df = validate(df, schema_name)

    return df


# ===========================================================================
# JSONL I/O (for LLM call logs and narratives)
# ===========================================================================


def append_jsonl(record: dict, path: Path | str) -> None:
    """
    Append a single JSON record to a JSONL file (creates file if not exists).

    Used for: data/runs/<run_id>/llm_calls.jsonl, case_narratives.jsonl.
    Not schema-validated (JSONL is used for LLM audit logs, not structured data).
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    """Read all records from a JSONL file. Returns empty list if file not found."""
    import json

    path = Path(path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ===========================================================================
# YAML helpers (for config read/write)
# ===========================================================================


def read_yaml(path: Path | str) -> dict:
    """Read a YAML config file and return as a dict."""
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict, path: Path | str) -> None:
    """
    Atomically write a dict to a YAML config file.

    Used by: CalibrationEngine.store_to_yaml(), Module 4B propose_mutation.
    """
    import yaml

    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path_str = tempfile.mkstemp(dir=path.parent, suffix=".yaml.tmp")
    tmp_path = Path(tmp_path_str)
    try:
        os.close(tmp_fd)
        with tmp_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


# ===========================================================================
# Run directory helpers
# ===========================================================================


def get_run_dir(base_data_dir: Path | str, run_id: str) -> Path:
    """Return the path for a simulation run directory (does not create it)."""
    return Path(base_data_dir) / "runs" / run_id


def ensure_run_dir(
    base_data_dir: Path | str,
    run_id: str,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Create the run directory for run_id.

    Args:
        base_data_dir: Root data directory (typically 'data/').
        run_id: Unique run identifier.
        overwrite: If False and directory already exists, raises FileExistsError
                   (implements the SRS Section 9 duplicate run_id protection).

    Returns:
        Path to the created run directory.
    """
    run_dir = get_run_dir(base_data_dir, run_id)

    if run_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Use --overwrite to intentionally overwrite an existing run, "
            "or choose a different run_id / seed combination."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
