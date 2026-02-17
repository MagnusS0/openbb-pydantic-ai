"""Progressive disclosure toolset for pydantic-ai toolsets."""

from __future__ import annotations

import html
import json
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

from pydantic import BaseModel, Field
from pydantic_ai import ToolReturn
from pydantic_ai.exceptions import CallDeferred, ModelRetry
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

if TYPE_CHECKING:
    from pydantic_ai import ToolsetTool

ToolsetSource = AbstractToolset[Any] | tuple[str, AbstractToolset[Any]]

_XML_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_MISSING_TOOL_GUIDANCE = (
    "Run list_tools or search_tools first, then get_tool_schema before calling tools."
)


class _ToolCallSpec(BaseModel):
    """Structured schema for a single call_tools entry."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


_DEFAULT_INSTRUCTIONS = """\
You have access to tools via progressive disclosure.
Use these meta-tools to discover and execute tools on demand:

- `list_tools(group=None)` — list tools grouped by source (optional group filter)
- `search_tools(query, group=None)` — search tools by name/description
  (optional group filter)
- `get_tool_schema(tool_names)` — inspect one or more tool schemas
  (XML-wrapped JSON objects)
- `call_tools(calls)` — execute one or more tools

Use the workflow: discover -> inspect -> execute.
Always pass `arguments` as an object/dictionary, never as a JSON string.
Always pass `calls` as a non-empty list, even for a single tool call.

Correct:
```
call_tools(calls=[{"tool_name":"chart","arguments":{"symbol":"SILJ"}}])
```

Incorrect:
```
call_tools(calls=[
  {"tool_name": "chart", "arguments": "{\\"symbol\\": \\"SILJ\\"}"}
])
```

Batch multiple tools in one call when possible:
```
call_tools(calls=[
  {"tool_name": "tool_a", "arguments": {"symbol": "AAPL"}},
  {"tool_name": "tool_b", "arguments": {"symbol": "MSFT"}}
])
```

You can reduce token usage by filtering tools by group, e.g.:
`list_tools(group="openbb_viz_tools")`

<available_tool_groups>
{tool_groups}
</available_tool_groups>
"""


@dataclass(slots=True, frozen=True)
class _RegisteredTool:
    """Internal metadata for a discovered tool."""

    name: str
    description: str
    schema: dict[str, Any]
    source_id: str
    source_toolset: AbstractToolset[Any] | None = None
    tool_handle: ToolsetTool[Any] | None = None


class ToolDiscoveryToolset(FunctionToolset[Any]):
    """Expose tools through progressive-disclosure meta-tools."""

    def __init__(
        self,
        *,
        toolsets: Sequence[ToolsetSource] | None = None,
        group_descriptions: dict[str, str] | None = None,
        id: str | None = None,
        instruction_template: str | None = None,
        exclude_meta_tools: set[str] | None = None,
    ) -> None:
        try:
            super().__init__(id=id)
        except TypeError:
            super().__init__()

        if instruction_template and "{tool_groups}" not in instruction_template:
            raise ValueError(
                "instruction_template must include the '{tool_groups}' placeholder."
            )

        valid_meta_tools = {
            "list_tools",
            "search_tools",
            "get_tool_schema",
            "call_tools",
        }
        self._exclude_meta_tools = set(exclude_meta_tools or set())
        invalid = self._exclude_meta_tools - valid_meta_tools
        if invalid:
            raise ValueError(
                f"Unknown meta-tools: {sorted(invalid)}. "
                f"Valid names: {sorted(valid_meta_tools)}"
            )
        if "call_tools" in self._exclude_meta_tools:
            warnings.warn(
                "'call_tools' is excluded; discovered tools cannot be executed.",
                UserWarning,
                stacklevel=2,
            )

        self._instruction_template = instruction_template or _DEFAULT_INSTRUCTIONS
        self._group_descriptions = group_descriptions or {}
        self._registry: dict[str, _RegisteredTool] = {}
        self._pending_toolsets: list[tuple[str, AbstractToolset[Any]]] = []
        self._known_group_ids: set[str] = set()
        self._resolved = False

        for idx, source in enumerate(toolsets or ()):
            if isinstance(source, tuple):
                source_id, toolset = source
            else:
                toolset = source
                source_id = str(getattr(toolset, "id", None) or f"toolset_{idx}")

            self._pending_toolsets.append((source_id, toolset))
            self._known_group_ids.add(source_id)

        self._register_meta_tools()

    def _register_tool(self, registered_tool: _RegisteredTool) -> None:
        existing = self._registry.get(registered_tool.name)
        if existing is not None:
            raise ValueError(
                f"Duplicate tool name '{registered_tool.name}' discovered in "
                f"'{existing.source_id}' and '{registered_tool.source_id}'. "
                "Tool names must be unique."
            )
        self._registry[registered_tool.name] = registered_tool
        self._known_group_ids.add(registered_tool.source_id)

    async def _resolve_pending(self, ctx: RunContext[Any]) -> None:
        if self._resolved:
            return
        self._resolved = True

        pending = list(self._pending_toolsets)
        self._pending_toolsets.clear()
        for source_id, toolset in pending:
            try:
                tools = await toolset.get_tools(ctx)
            except Exception as exc:
                warnings.warn(
                    f"Could not resolve toolset '{source_id}': {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for tool_name, tool_handle in tools.items():
                try:
                    tool_def = tool_handle.tool_def
                except AttributeError as exc:
                    raise RuntimeError(
                        "Unsupported ToolsetTool shape from pydantic-ai: "
                        "expected 'tool_def' attribute on tool handles."
                    ) from exc

                parameters_schema = tool_def.parameters_json_schema or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }
                try:
                    self._register_tool(
                        _RegisteredTool(
                            name=tool_name,
                            description=tool_def.description or "",
                            schema=parameters_schema,
                            source_id=source_id,
                            source_toolset=toolset,
                            tool_handle=tool_handle,
                        )
                    )
                except ValueError:
                    warnings.warn(
                        f"Duplicate tool name '{tool_name}' in source "
                        f"'{source_id}' conflicts with existing registration. "
                        "Skipping this tool.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

    def _register_meta_tools(self) -> None:
        if "list_tools" not in self._exclude_meta_tools:

            @self.tool
            async def list_tools(ctx: RunContext[Any], group: str | None = None) -> str:
                """List available tools and short descriptions.

                Args:
                    group: Optional exact group id to filter tools.

                Returns:
                    Markdown grouped by tool source.
                """
                return await self._list_tools_impl(ctx, group=group)

        if "search_tools" not in self._exclude_meta_tools:

            @self.tool
            async def search_tools(
                ctx: RunContext[Any], query: str, group: str | None = None
            ) -> str:
                """Search tools by keyword in name/description.

                Args:
                    query: Case-insensitive keyword to match.
                    group: Optional exact group id to filter tools.

                Returns:
                    Markdown grouped by tool source.
                """
                return await self._search_tools_impl(ctx, query, group=group)

        if "get_tool_schema" not in self._exclude_meta_tools:

            @self.tool
            async def get_tool_schema(
                ctx: RunContext[Any],
                tool_names: list[str],
            ) -> str:
                """Return full schema metadata for one or more tools.

                Args:
                    tool_names: List of exact tool names from `list_tools`.

                Returns:
                    XML-like blocks where each block wraps a compact JSON object.
                """
                return await self._get_tool_schema_impl(ctx, tool_names)

        if "call_tools" not in self._exclude_meta_tools:

            @self.tool(retries=2)
            async def call_tools(
                ctx: RunContext[Any],
                calls: Annotated[list[_ToolCallSpec], Field(min_length=1)],
            ) -> Any:
                """Execute one or more tools.

                Args:
                    calls: List of invocation objects. Each entry must have
                        `tool_name` (str) and optionally `arguments` (object).
                        Examples:
                        [
                            {"tool_name": "widget_a", "arguments": {"symbol": "AAPL"}},
                            {"tool_name": "widget_b", "arguments": {"symbol": "MSFT"}}
                        ]

                Returns:
                    Deferred requests (for deferred tools), a native
                    `ToolReturn` for single-call metadata-preserving tool
                    responses, or a markdown summary grouped by tool invocation.
                """
                return await self._call_tools_impl(
                    ctx, [call.model_dump() for call in calls]
                )

    @staticmethod
    def _json_compact(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _tool_result_to_text(value: Any) -> str:
        if isinstance(value, ToolReturn):
            # ToolReturn metadata is app-only; only expose the return value in
            # textual summaries to avoid leaking metadata into model context.
            return ToolDiscoveryToolset._tool_result_to_text(value.return_value)
        if value is None:
            return "null"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else "(empty)"
        try:
            return ToolDiscoveryToolset._json_compact(value)
        except TypeError:
            return str(value)

    @staticmethod
    def _schema_block(tag_name: str, payload: dict[str, Any]) -> str:
        payload_text = ToolDiscoveryToolset._json_compact(payload)
        if _XML_TAG_NAME_RE.match(tag_name):
            return f"<{tag_name}>\n{payload_text}\n</{tag_name}>"
        escaped = html.escape(tag_name, quote=True)
        return f'<tool name="{escaped}">\n{payload_text}\n</tool>'

    @staticmethod
    def _raise_missing_tools(missing_tool_names: Sequence[str]) -> NoReturn:
        missing_display = ", ".join(missing_tool_names[:20])
        raise ModelRetry(
            f"Tool(s) not found: {missing_display}. {_MISSING_TOOL_GUIDANCE}"
        )

    @staticmethod
    def _raise_missing_tool(tool_name: str) -> NoReturn:
        raise ModelRetry(f"Tool '{tool_name}' not found. {_MISSING_TOOL_GUIDANCE}")

    def _get_registered_tool(self, tool_name: str) -> _RegisteredTool:
        if tool_name not in self._registry:
            self._raise_missing_tool(tool_name)
        return self._registry[tool_name]

    def _format_tools_markdown(
        self,
        items: list[tuple[str, _RegisteredTool]],
        *,
        group: str | None = None,
    ) -> str:
        if not items:
            if group:
                return f"No tools found for group '{group}'."
            return "No tools found."

        grouped: dict[str, list[tuple[str, str]]] = {}
        for name, tool in items:
            grouped.setdefault(tool.source_id, []).append(
                (name, tool.description[:200])
            )

        lines: list[str] = []
        for group_id in sorted(grouped):
            tools = grouped[group_id]
            lines.append(f"# {group_id}")
            lines.append(f"count: {len(tools)}")
            for tool_name, tool_desc in tools:
                if tool_desc:
                    lines.append(f"- {tool_name}: {tool_desc}")
                else:
                    lines.append(f"- {tool_name}")
            lines.append("")
        return "\n".join(lines).strip()

    async def _list_tools_impl(
        self,
        ctx: RunContext[Any],
        *,
        group: str | None = None,
    ) -> str:
        await self._resolve_pending(ctx)
        items = [
            (name, tool)
            for name, tool in sorted(self._registry.items())
            if group is None or tool.source_id == group
        ]
        return self._format_tools_markdown(items, group=group)

    async def _search_tools_impl(
        self,
        ctx: RunContext[Any],
        query: str,
        *,
        group: str | None = None,
    ) -> str:
        await self._resolve_pending(ctx)
        q = query.strip().lower()
        if not q:
            return "No tools found."
        tokens = [token for token in q.replace("_", " ").split() if token]
        items = [
            (name, tool)
            for name, tool in sorted(self._registry.items())
            if (
                (group is None or tool.source_id == group)
                and (
                    q in (haystack := f"{name} {tool.description}".lower())
                    or all(token in haystack for token in tokens)
                )
            )
        ]
        return self._format_tools_markdown(items, group=group)

    async def _get_tool_schema_impl(
        self,
        ctx: RunContext[Any],
        tool_names: list[str],
    ) -> str:
        await self._resolve_pending(ctx)

        names = [name for name in tool_names if isinstance(name, str) and name]

        # Preserve order while removing duplicates.
        deduped_names = list(dict.fromkeys(names))
        if not deduped_names:
            raise ModelRetry("`tool_names` must contain at least one tool name.")

        missing = [name for name in deduped_names if name not in self._registry]
        if missing:
            self._raise_missing_tools(missing)

        blocks: list[str] = []
        for name in deduped_names:
            tool = self._registry[name]
            payload = {
                "description": tool.description,
                "group": tool.source_id,
                "parameters": tool.schema,
            }
            blocks.append(self._schema_block(tool.name, payload))
        return "\n".join(blocks)

    def _is_deferred_tool(self, tool: _RegisteredTool) -> bool:
        """Check if a tool will be deferred based on its kind attribute."""
        if tool.tool_handle is None:
            return False
        tool_kind = getattr(tool.tool_handle.tool_def, "kind", None)
        return tool_kind == "external"

    async def _call_tool_impl(
        self,
        ctx: RunContext[Any],
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self._resolve_pending(ctx)
        resolved_tool_name = tool_name
        tool = self._get_registered_tool(resolved_tool_name)

        if arguments is None:
            args = {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            raise ModelRetry("Tool arguments must be an object/dictionary.")

        if tool.source_toolset is not None and tool.tool_handle is not None:
            tool_kind = getattr(tool.tool_handle.tool_def, "kind", None)
            if tool_kind == "external":
                try:
                    return await tool.source_toolset.call_tool(
                        resolved_tool_name,
                        args,
                        ctx,
                        tool.tool_handle,
                    )
                except NotImplementedError as err:
                    # ExternalToolset from pydantic-ai intentionally raises
                    # here.  Raise CallDeferred so the agent graph pauses the
                    # run (deferred semantics) instead of treating the result
                    # as a regular tool return which would corrupt message
                    # history.
                    raise CallDeferred(
                        metadata={
                            "tool_name": resolved_tool_name,
                            "arguments": args,
                        }
                    ) from err

            return await tool.source_toolset.call_tool(
                resolved_tool_name,
                args,
                ctx,
                tool.tool_handle,
            )

        raise RuntimeError(f"Tool '{tool_name}' has no executable callable configured.")

    async def _call_tools_impl(
        self,
        ctx: RunContext[Any],
        calls: list[dict[str, Any]] | dict[str, Any],
    ) -> Any:
        """Execute tools, preserving deferred semantics and single-call ToolReturn."""
        normalized_calls: list[dict[str, Any]]
        if isinstance(calls, dict):
            normalized_calls = [calls]
        elif isinstance(calls, list):
            normalized_calls = calls
        else:
            raise ModelRetry("`calls` must be an object or a list of objects.")

        if not normalized_calls:
            raise ModelRetry("`calls` must contain at least one call.")

        for i, entry in enumerate(normalized_calls):
            if not isinstance(entry, dict) or "tool_name" not in entry:
                raise ModelRetry(f"Entry {i} must be an object with a `tool_name` key.")

        # Pre-classify tools to detect mixed deferred/immediate before execution
        await self._resolve_pending(ctx)
        deferred_candidates: list[str] = []
        immediate_candidates: list[str] = []

        for entry in normalized_calls:
            tool_name = entry["tool_name"]
            tool = self._get_registered_tool(tool_name)

            if self._is_deferred_tool(tool):
                deferred_candidates.append(tool_name)
            else:
                immediate_candidates.append(tool_name)

        # Detect mixed batch before executing any tools
        if deferred_candidates and immediate_candidates:
            deferred_display = ", ".join(sorted(set(deferred_candidates))[:12])
            immediate_display = ", ".join(sorted(set(immediate_candidates))[:12])
            raise ModelRetry(
                "The `call_tools` batch mixed deferred tools "
                "and immediate tools. "
                f"Deferred tools: {deferred_display}. "
                f"Immediate tools: {immediate_display}. "
                "Split them into separate `call_tools` calls: "
                "first run immediate tools together, "
                "then run deferred tools together. "
                "Refer to `CallDeferred` for deferred semantics."
            )

        # Execute tools (all deferred or all immediate at this point)
        results: list[dict[str, Any]] = []
        deferred_calls: list[dict[str, Any]] = []

        for entry in normalized_calls:
            tool_name = entry["tool_name"]
            arguments = entry.get("arguments")
            try:
                result = await self._call_tool_impl(ctx, tool_name, arguments)
            except CallDeferred as exc:
                deferred_calls.append(
                    exc.metadata or {"tool_name": tool_name, "arguments": arguments}
                )
            else:
                results.append({"tool_name": tool_name, "result": result})

        if deferred_calls:
            # Re-raise CallDeferred so the agent graph pauses the run
            # with proper deferred semantics.  The metadata carries the
            # actual tool calls so the adapter can reconstruct them.
            raise CallDeferred(metadata={"deferred_calls": deferred_calls})

        # Preserve native ToolReturn semantics for single-call execution so
        # metadata (e.g., chart/html artifacts) remains available to the app.
        if len(results) == 1:
            single_result = results[0]["result"]
            if isinstance(single_result, ToolReturn):
                return single_result

        lines: list[str] = ["# Results"]
        for entry in results:
            lines.append("")
            lines.append(f"## {entry['tool_name']}")
            lines.append(self._tool_result_to_text(entry["result"]))
        return "\n".join(lines)

    async def get_instructions(self, ctx: RunContext[Any]) -> str | None:
        await self._resolve_pending(ctx)
        return self.render_instructions()

    def render_instructions(self) -> str | None:
        """Render static discovery instructions for prompt injection."""
        if not self._registry and not self._known_group_ids:
            return None

        groups: dict[str, list[str]] = {}
        for tool in self._registry.values():
            groups.setdefault(tool.source_id, []).append(tool.name)

        unresolved = sorted(self._known_group_ids - groups.keys())
        lines: list[str] = []

        for group_id, names in sorted(groups.items()):
            description = self._group_descriptions.get(group_id, "")
            if description:
                lines.append(
                    (
                        f'<group name="{group_id}" '
                        f'description="{description}" '
                        f'tool_count="{len(names)}">'
                    )
                )
            else:
                lines.append(f'<group name="{group_id}" tool_count="{len(names)}">')
            for tool_name in sorted(names):
                tool_desc = self._registry[tool_name].description[:120]
                lines.append(f'  <tool name="{tool_name}">{tool_desc}</tool>')
            lines.append("</group>")

        for group_id in unresolved:
            description = self._group_descriptions.get(group_id, "")
            if description:
                lines.append(
                    (
                        f'<group name="{group_id}" '
                        f'description="{description}" '
                        'tool_count="pending" />'
                    )
                )
            else:
                lines.append(f'<group name="{group_id}" tool_count="pending" />')

        tool_groups = (
            "\n".join(lines) if lines else '<group name="none" tool_count="0" />'
        )
        return self._instruction_template.replace("{tool_groups}", tool_groups)

    @property
    def discovered_tools(self) -> dict[str, str]:
        """Snapshot of currently discovered tool names and descriptions."""
        return {name: tool.description for name, tool in sorted(self._registry.items())}
