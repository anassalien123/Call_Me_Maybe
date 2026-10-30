# ABOUTME: Helpers to load and save the JSON files consumed/produced by
# ABOUTME: the project. All functions raise a ``ProjectIOError`` with a
# ABOUTME: clear message instead of crashing.

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter, ValidationError

from src.schema import FunctionCallResult, FunctionDefinition, PromptEntry


class ProjectIOError(Exception):
    """Raised whenever an input/output file cannot be read or written
    safely."""


def _read_json_array(path: Path) -> list[dict[str, object]]:
    """Read *path* and return its content as a list of JSON objects.

    Raises:
        ProjectIOError: if the file is missing, unreadable, not valid JSON,
            or does not contain a top-level JSON array.
    """
    if not path.exists():
        raise ProjectIOError(f"Input file not found: {path}")
    if not path.is_file():
        raise ProjectIOError(f"Expected a file but found a directory: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            content = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectIOError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise ProjectIOError(f"Could not read {path}: {exc}") from exc

    if not isinstance(content, list):
        raise ProjectIOError(
            f"Expected a JSON array at the top level of {path}"
        )
    return content


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate ``functions_definition.json``."""
    raw_items = _read_json_array(path)
    try:
        adapter = TypeAdapter(list[FunctionDefinition])
        validated = adapter.validate_python(raw_items)
        return cast("list[FunctionDefinition]", validated)
    except ValidationError as exc:
        raise ProjectIOError(
            f"{path} does not match the expected schema: {exc}"
        ) from exc


def load_prompts(path: Path) -> list[PromptEntry]:
    """Load and validate ``function_calling_tests.json``."""
    raw_items = _read_json_array(path)
    try:
        adapter = TypeAdapter(list[PromptEntry])
        validated = adapter.validate_python(raw_items)
        return cast("list[PromptEntry]", validated)
    except ValidationError as exc:
        raise ProjectIOError(
            f"{path} does not match the expected schema: {exc}"
        ) from exc


def save_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write *results* as a JSON array to *path*, creating parent
    directories as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump() for result in results]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise ProjectIOError(
            f"Could not write output file {path}: {exc}"
        ) from exc
