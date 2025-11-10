"""Utility functions for OpenBB Pydantic AI UI adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ._config import (
    MAX_ARG_DISPLAY_CHARS,
    MAX_ARG_PREVIEW_ITEMS,
)
from ._serializers import ContentSerializer


def hash_tool_call(function: str, input_arguments: dict[str, Any]) -> str:
    """Generate a deterministic hash-based ID for a tool call.

    This creates a unique identifier by hashing the function name and arguments,
    ensuring consistent tool call IDs across message history and deferred results.

    Parameters
    ----------
    function : str
        The name of the function/tool being called
    input_arguments : dict[str, Any]
        The arguments passed to the tool

    Returns
    -------
    str
        A string combining the function name with a 16-character hash digest
    """
    payload = json.dumps(
        {"function": function, "input_arguments": input_arguments},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{function}_{digest[:16]}"


def normalize_args(args: Any) -> dict[str, Any]:
    """Normalize tool call arguments to a dictionary."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    return {}


def get_str(mapping: Mapping[str, Any], *keys: str) -> str | None:
    """Return the first string value found for the given keys."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return None


def get_str_list(mapping: Mapping[str, Any], *keys: str) -> list[str] | None:
    """Return the first list of strings (or single string) found for the keys."""
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            items = [item for item in value if isinstance(item, str)]
            if items:
                return items
    return None


def _truncate(value: str, max_chars: int = 160) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _json_dump(value: Any) -> str:
    return ContentSerializer.to_json(value)


def format_arg_value(
    value: Any,
    *,
    max_chars: int = MAX_ARG_DISPLAY_CHARS,
    max_items: int = MAX_ARG_PREVIEW_ITEMS,
) -> str:
    """Summarize nested structures so reasoning details stay readable."""

    if isinstance(value, str):
        return _truncate(value, max_chars)

    if isinstance(value, (int, float, bool)) or value is None:
        return _json_dump(value)

    if isinstance(value, Mapping):
        keys = list(value.keys())
        preview_keys = keys[:max_items]
        preview = {k: value[k] for k in preview_keys}
        suffix = "..." if len(keys) > max_items else ""
        return _truncate(
            f"dict(keys={preview_keys}{suffix}, sample={_json_dump(preview)})",
            max_chars,
        )

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        seq = list(value)
        preview = seq[:max_items]
        suffix = "..." if len(seq) > max_items else ""
        return _truncate(
            f"list(len={len(seq)}{suffix}, sample={_json_dump(preview)})",
            max_chars,
        )

    return _truncate(_json_dump(value), max_chars)


def format_args(args: Mapping[str, Any]) -> dict[str, str]:
    """Format a mapping of arguments into readable key/value strings."""

    formatted: dict[str, str] = {}
    for key, value in args.items():
        formatted[key] = format_arg_value(value)
    return formatted
