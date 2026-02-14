"""Formatting helpers for discovery/meta tool call and result display."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from openbb_pydantic_ai._config import CONTENT_PREVIEW_MAX_CHARS
from openbb_pydantic_ai._serializers import parse_json, to_string
from openbb_pydantic_ai._utils import format_arg_value


def _normalize_tool_description(value: Any) -> str:
    text = to_string(value) or ""
    return " ".join(text.split())


def _preview_names(names: Sequence[str], limit: int = 12) -> str:
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview += ", ..."
    return preview


def _preview_tool_list(tools: Sequence[tuple[str, str]], limit: int = 12) -> str:
    lines = [f"{name}: {desc}" if desc else name for name, desc in tools[:limit]]
    remaining = len(tools) - limit
    if remaining > 0:
        lines.append(f"... and {remaining} more")
    return "\n".join(lines)


def _parse_markdown_tool_listing(content: str) -> list[tuple[str, str]]:
    tools: list[tuple[str, str]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        name, sep, desc = body.partition(":")
        tool_name = name.strip()
        if not tool_name:
            continue
        tool_desc = desc.strip() if sep else ""
        tools.append((tool_name, tool_desc))
    return tools


def _parse_schema_blocks(content: str) -> list[tuple[str, dict[str, Any]]]:
    blocks: list[tuple[str, dict[str, Any]]] = []

    # <tool_name> ...json... </tool_name>
    for match in re.finditer(
        r"<([A-Za-z_][A-Za-z0-9_.-]*)>\s*(.*?)\s*</\1>",
        content,
        flags=re.DOTALL,
    ):
        tag_name = match.group(1)
        payload = parse_json(match.group(2))
        if isinstance(payload, dict):
            blocks.append((tag_name, payload))

    # <tool name="tool_name"> ...json... </tool>
    for match in re.finditer(
        r'<tool name="([^"]+)">\s*(.*?)\s*</tool>',
        content,
        flags=re.DOTALL,
    ):
        tag_name = match.group(1)
        payload = parse_json(match.group(2))
        if isinstance(payload, dict):
            blocks.append((tag_name, payload))

    return blocks


def _tool_list_details(
    entries: Sequence[tuple[str, str]],
    *,
    label: str,
    count_label: str,
) -> dict[str, str]:
    if not entries:
        return {
            count_label: "0",
            label: "(none)",
        }

    return {
        count_label: str(len(entries)),
        label: _preview_tool_list(entries),
    }


def _add_parameter_details(details: dict[str, str], payload: Mapping[str, Any]) -> None:
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        return

    properties = parameters.get("properties")
    required = parameters.get("required")
    property_names = (
        [str(key) for key in properties.keys()]
        if isinstance(properties, Mapping)
        else []
    )
    required_count = len(required) if isinstance(required, list) else 0
    details["Parameter count"] = str(len(property_names))
    details["Required count"] = str(required_count)
    if property_names:
        details["Parameters"] = _preview_names(property_names)


def _schema_overview(names: Sequence[str], schema_count: int) -> dict[str, str]:
    details = {"Schema count": str(schema_count)}
    if names:
        details["Tools"] = _preview_names(names)
    return details


def _single_tool_schema_details(
    name: str,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    details: dict[str, str] = {"Name": name}

    group = payload.get("group")
    description = payload.get("description")
    if isinstance(group, str) and group:
        details["Group"] = group
    if isinstance(description, str) and description:
        details["Description"] = _normalize_tool_description(description)

    _add_parameter_details(details, payload)
    return details


def _discovery_tool_labels(tool_name: str) -> tuple[str, str]:
    if tool_name == "search_tools":
        return ("Matches", "Match count")
    return ("Tools", "Tool count")


def _extract_discovery_tool_entries(
    content: Mapping[str, Any],
) -> list[tuple[str, str]]:
    return sorted(
        (str(name), _normalize_tool_description(desc)) for name, desc in content.items()
    )


def _format_discovery_listing_result(
    tool_name: str,
    content: Any,
) -> dict[str, str] | None:
    if tool_name not in {"list_tools", "search_tools"}:
        return None

    label, count_label = _discovery_tool_labels(tool_name)
    if isinstance(content, Mapping):
        entries = _extract_discovery_tool_entries(content)
        return _tool_list_details(entries, label=label, count_label=count_label)

    if isinstance(content, str):
        entries = _parse_markdown_tool_listing(content)
        return _tool_list_details(entries, label=label, count_label=count_label)

    return None


def _schema_details_from_payload(payload: Mapping[str, Any]) -> dict[str, str] | None:
    multi_tools = payload.get("tools")
    if isinstance(multi_tools, list):
        tool_dicts = [item for item in multi_tools if isinstance(item, Mapping)]
        names = [
            str(item.get("name"))
            for item in tool_dicts
            if isinstance(item.get("name"), str) and item.get("name")
        ]
        return _schema_overview(names, len(tool_dicts))

    details: dict[str, str] = {}
    name = payload.get("name")
    group = payload.get("group")
    description = payload.get("description")
    if isinstance(name, str) and name:
        details["Name"] = name
    if isinstance(group, str) and group:
        details["Group"] = group
    if isinstance(description, str) and description:
        details["Description"] = _normalize_tool_description(description)

    _add_parameter_details(details, payload)
    return details or None


def _format_tool_schema_result(content: Any) -> dict[str, str] | None:
    if not isinstance(content, str):
        return None

    blocks = _parse_schema_blocks(content)
    if blocks:
        if len(blocks) == 1:
            tag_name, payload = blocks[0]
            return _single_tool_schema_details(tag_name, payload)

        names = [name for name, _ in blocks if name]
        return _schema_overview(names, len(blocks))

    parsed = parse_json(content)
    if not isinstance(parsed, Mapping):
        return None

    return _schema_details_from_payload(parsed)


def _format_call_tools_result(content: Any) -> dict[str, str] | None:
    if isinstance(content, list):
        tool_results = [
            entry
            for entry in content
            if isinstance(entry, dict) and "tool_name" in entry
        ]
        if not tool_results:
            return None

        lines: list[str] = []
        for entry in tool_results[:12]:
            name = entry.get("tool_name", "?")
            result = entry.get("result")
            result_preview = format_arg_value(
                result,
                max_chars=CONTENT_PREVIEW_MAX_CHARS,
            )
            lines.append(f"{name}: {result_preview}")

        remaining = len(tool_results) - 12
        if remaining > 0:
            lines.append(f"... and {remaining} more")

        return {
            "Result count": str(len(tool_results)),
            "Results": "\n".join(lines),
        }

    if isinstance(content, str):
        result_names: list[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("## "):
                name = line[3:].strip()
                if name:
                    result_names.append(name)

        if result_names:
            return {
                "Result count": str(len(result_names)),
                "Results": _preview_names(result_names),
            }

    return None


def _format_discovery_meta_result(
    tool_name: str, content: Any
) -> dict[str, str] | None:
    listing_result = _format_discovery_listing_result(tool_name, content)
    if listing_result is not None:
        return listing_result

    if tool_name == "get_tool_schema":
        return _format_tool_schema_result(content)

    if tool_name == "call_tools":
        return _format_call_tools_result(content)

    return None


def _format_meta_tool_call_args(
    tool_name: str, args: dict[str, Any]
) -> dict[str, str] | None:
    """Format discovery meta-tool call args for readable reasoning steps."""
    if tool_name == "call_tools":
        calls = args.get("calls")
        if not isinstance(calls, list) or not calls:
            return None
        lines: list[str] = []
        for entry in calls[:12]:
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool_name", "?")
            entry_args = entry.get("arguments", {})
            if isinstance(entry_args, dict) and entry_args:
                params = ", ".join(
                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                    for k, v in list(entry_args.items())[:3]
                )
                lines.append(f"{name}({params})")
            else:
                lines.append(name)
        remaining = len(calls) - 12
        if remaining > 0:
            lines.append(f"... and {remaining} more")
        return {
            "Tool count": str(len(calls)),
            "Tools": "\n".join(lines),
        }

    if tool_name == "get_tool_schema":
        tool_names = args.get("tool_names")
        if isinstance(tool_names, list) and tool_names:
            names = [str(name) for name in tool_names if isinstance(name, str)]
            if names:
                return {"Tools": _preview_names(names)}

    return None
