"""Progressive disclosure toolset for pydantic-ai toolsets."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

if TYPE_CHECKING:
    from pydantic_ai import ToolsetTool

ToolsetSource = AbstractToolset[Any] | tuple[str, AbstractToolset[Any]]


_DEFAULT_INSTRUCTIONS = """\
You have access to tools via progressive disclosure.
Use these meta-tools to discover and execute tools on demand:

- `list_tools()` — list all available tool names and short descriptions
- `search_tools(query)` — filter tools by name/description
- `get_tool_schema(tool_names)` — inspect one or more tool schemas
- `call_tools(calls)` — execute one or more tools

Use the workflow: discover -> inspect -> execute.
Always pass `arguments` as an object/dictionary, never as a JSON string.

Correct:
```
call_tools(calls=[
  {"tool_name": "chart", "arguments": {"symbol": "SILJ"}}
])
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

<available_tool_groups>
{tool_groups}
</available_tool_groups>
"""


@dataclass(slots=True)
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

    def _register_meta_tools(self) -> None:
        if "list_tools" not in self._exclude_meta_tools:

            @self.tool
            async def list_tools(ctx: RunContext[Any]) -> dict[str, str]:
                """List available tools and short descriptions.

                Returns:
                    Mapping of tool name to one-line description.
                """
                return await self._list_tools_impl(ctx)

        if "search_tools" not in self._exclude_meta_tools:

            @self.tool
            async def search_tools(ctx: RunContext[Any], query: str) -> dict[str, str]:
                """Search tools by keyword in name/description.

                Args:
                    query: Case-insensitive keyword to match.

                Returns:
                    Mapping of matching tool names to descriptions.
                """
                return await self._search_tools_impl(ctx, query)

        if "get_tool_schema" not in self._exclude_meta_tools:

            @self.tool
            async def get_tool_schema(
                ctx: RunContext[Any],
                tool_names: list[str],
            ) -> str:
                """Return full schema metadata for one or more tools.

                Args:
                    tool_names: List of exact tool names from `list_tools`.
                """
                return await self._get_tool_schema_impl(ctx, tool_names)

        if "call_tools" not in self._exclude_meta_tools:

            @self.tool
            async def call_tools(
                ctx: RunContext[Any],
                calls: list[dict[str, Any]],
            ) -> Any:
                """Execute one or more tools.

                Args:
                    calls: List of tool invocations. Each entry must have
                        `tool_name` (str) and optionally `arguments` (object).
                        Example: [
                            {"tool_name": "widget_a", "arguments": {"symbol": "AAPL"}},
                            {"tool_name": "widget_b", "arguments": {"symbol": "MSFT"}}
                        ]
                """
                return await self._call_tools_impl(ctx, calls)

    async def _list_tools_impl(self, ctx: RunContext[Any]) -> dict[str, str]:
        await self._resolve_pending(ctx)
        return {
            name: tool.description[:200]
            for name, tool in sorted(self._registry.items())
        }

    async def _search_tools_impl(
        self, ctx: RunContext[Any], query: str
    ) -> dict[str, str]:
        await self._resolve_pending(ctx)
        q = query.strip().lower()
        if not q:
            return {}
        tokens = [token for token in q.replace("_", " ").split() if token]
        return {
            name: tool.description[:200]
            for name, tool in sorted(self._registry.items())
            if (
                q in (haystack := f"{name} {tool.description}".lower())
                or all(token in haystack for token in tokens)
            )
        }

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
            raise ValueError("`tool_names` must contain at least one tool name.")

        missing = [name for name in deduped_names if name not in self._registry]
        if missing:
            available = ", ".join(sorted(self._registry)[:20])
            missing_display = ", ".join(missing[:20])
            raise ValueError(
                f"Tool(s) not found: {missing_display}. "
                f"Some available tools: {available}"
            )

        schemas = [
            {
                "name": self._registry[name].name,
                "description": self._registry[name].description,
                "group": self._registry[name].source_id,
                "parameters": self._registry[name].schema,
            }
            for name in deduped_names
        ]

        return json.dumps(
            {"tools": schemas, "count": len(schemas)},
            indent=2,
        )

    async def _call_tool_impl(
        self,
        ctx: RunContext[Any],
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self._resolve_pending(ctx)
        resolved_tool_name = tool_name
        tool = self._registry.get(resolved_tool_name)
        if tool is None:
            available = ", ".join(sorted(self._registry)[:20])
            raise ValueError(
                f"Tool '{tool_name}' not found. Some available tools: {available}"
            )

        if arguments is None:
            args = {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            raise ValueError("Tool arguments must be an object/dictionary.")

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
                except NotImplementedError:
                    # ExternalToolset from pydantic-ai intentionally raises here.
                    # Return a deferred request so the normal OpenBB deferred
                    # execution pipeline can handle it.
                    deferred = DeferredToolRequests()
                    deferred.calls.append(
                        ToolCallPart(
                            tool_name=resolved_tool_name,
                            args=args,
                        )
                    )
                    return deferred

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
        calls: list[dict[str, Any]],
    ) -> Any:
        """Execute multiple tools and merge deferred requests."""
        if not calls:
            raise ValueError("`calls` must be a non-empty list.")
        for i, entry in enumerate(calls):
            if not isinstance(entry, dict) or "tool_name" not in entry:
                raise ValueError(f"Entry {i} must be an object with a `tool_name` key.")

        results: list[dict[str, Any]] = []
        immediate_tool_names: list[str] = []
        deferred_tool_names: list[str] = []
        merged_deferred: DeferredToolRequests | None = None

        for entry in calls:
            tool_name = entry["tool_name"]
            arguments = entry.get("arguments")
            result = await self._call_tool_impl(ctx, tool_name, arguments)

            if isinstance(result, DeferredToolRequests):
                if merged_deferred is None:
                    merged_deferred = DeferredToolRequests()
                merged_deferred.calls.extend(result.calls)
                deferred_tool_names.append(tool_name)
            else:
                results.append({"tool_name": tool_name, "result": result})
                immediate_tool_names.append(tool_name)

        if merged_deferred is not None and results:
            deferred_display = ", ".join(sorted(set(deferred_tool_names))[:12])
            immediate_display = ", ".join(sorted(set(immediate_tool_names))[:12])
            raise ValueError(
                "The `call_tools` batch mixed deferred tools and immediate tools. "
                f"Deferred tools: {deferred_display}. "
                f"Immediate tools: {immediate_display}. "
                "Split them into separate `call_tools` calls: first run immediate "
                "tools together, then run deferred tools together."
            )

        if merged_deferred is not None:
            return merged_deferred
        if len(results) == 1:
            return results[0]["result"]
        return results

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
