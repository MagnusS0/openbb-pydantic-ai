"""Utility functions for OpenBB Pydantic AI UI adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from openbb_ai.models import (
    LlmClientFunctionCallResultMessage,
    Widget,
    WidgetCollection,
)

from openbb_pydantic_ai._config import (
    MAX_ARG_DISPLAY_CHARS,
    MAX_ARG_PREVIEW_ITEMS,
)
from openbb_pydantic_ai._serializers import parse_json, to_json


def iter_widget_collection(collection: WidgetCollection) -> Iterator[Widget]:
    """Iterate all widgets in a collection across priority groups."""
    for group in (collection.primary, collection.secondary, collection.extra):
        yield from group


def normalize_args(args: Any) -> dict[str, Any]:
    """Normalize tool call arguments to a dictionary."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        parsed = parse_json(args)
        if isinstance(parsed, dict):
            return parsed
    return {}


def extract_tool_call_id(message: LlmClientFunctionCallResultMessage) -> str:
    """Extract the tool_call_id from a result message or raise if missing."""

    extra_state = message.extra_state or {}
    if isinstance(extra_state, dict):
        extra_id = extra_state.get("tool_call_id")
        if isinstance(extra_id, str):
            return extra_id

        tool_calls = extra_state.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first = tool_calls[0]
            if isinstance(first, dict):
                tcid = first.get("tool_call_id")
                if isinstance(tcid, str):
                    return tcid

    raise ValueError(
        """
        `tool_call_id` is required for deferred tool results but was not found
        in message.extra_state
        """.strip()
    )


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
        return to_json(value)

    if isinstance(value, Mapping):
        keys = list(value.keys())
        preview_keys = keys[:max_items]
        preview = {k: value[k] for k in preview_keys}
        suffix = "..." if len(keys) > max_items else ""
        return _truncate(
            f"dict(keys={preview_keys}{suffix}, sample={to_json(preview)})",
            max_chars,
        )

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        seq = list(value)
        preview = seq[:max_items]
        suffix = "..." if len(seq) > max_items else ""
        return _truncate(
            f"list(len={len(seq)}{suffix}, sample={to_json(preview)})",
            max_chars,
        )

    return _truncate(to_json(value), max_chars)


def format_args(args: Mapping[str, Any]) -> dict[str, str]:
    """Format a mapping of arguments into readable key/value strings."""

    formatted: dict[str, str] = {}
    for key, value in args.items():
        formatted[key] = format_arg_value(value)
    return formatted
