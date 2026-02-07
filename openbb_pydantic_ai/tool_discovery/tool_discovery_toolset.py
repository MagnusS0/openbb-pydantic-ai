"""Progressive disclosure toolset for pydantic-ai toolsets."""

from __future__ import annotations

import inspect
import json
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai import _function_schema
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

if TYPE_CHECKING:
    from pydantic_ai import ToolsetTool

CallFn = Callable[[RunContext[Any], str, dict[str, Any]], Any | Awaitable[Any]]
ToolsetSource = AbstractToolset[Any] | tuple[str, AbstractToolset[Any]]


_DEFAULT_INSTRUCTIONS = """\
You have access to tools via progressive disclosure.
Use these meta-tools to discover and execute tools on demand:

1. `list_tools()` — list all available tool names and short descriptions
2. `search_tools(query)` — filter tools by name/description
3. `get_tool_schema(tool_name)` — inspect full JSON parameter schema
4. `call_tool(tool_name, arguments)` — execute the selected tool

Use the workflow: discover -> inspect -> execute.

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
    call_fn: CallFn | None = None


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
            "call_tool",
        }
        self._exclude_meta_tools = set(exclude_meta_tools or set())
        invalid = self._exclude_meta_tools - valid_meta_tools
        if invalid:
            raise ValueError(
                f"Unknown meta-tools: {sorted(invalid)}. "
                f"Valid names: {sorted(valid_meta_tools)}"
            )
        if "call_tool" in self._exclude_meta_tools:
            warnings.warn(
                "'call_tool' is excluded; discovered tools cannot be executed.",
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
                tool_def = getattr(tool_handle, "definition", None)
                if tool_def is None:
                    warnings.warn(
                        (
                            "Could not read definition for tool "
                            f"'{tool_name}' in '{source_id}'."
                        ),
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

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

    def discoverable_tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        group: str = "custom",
    ) -> Any:
        """Register a function behind the progressive-disclosure layer."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            default_name = getattr(f, "__name__", "tool")
            tool_name = name or default_name
            func_schema = _function_schema.function_schema(
                f,
                schema_generator=GenerateToolJsonSchema,
                takes_ctx=None,
                docstring_format="auto",
                require_parameter_descriptions=False,
            )
            tool_description = description or func_schema.description or ""

            async def _call(
                call_ctx: RunContext[Any], _: str, args: dict[str, Any]
            ) -> Any:
                return await func_schema.call(args, call_ctx)

            self._register_tool(
                _RegisteredTool(
                    name=tool_name,
                    description=tool_description,
                    schema=func_schema.json_schema,
                    source_id=group,
                    call_fn=_call,
                )
            )
            return f

        if func is None:
            return decorator
        return decorator(func)

    def register_tool(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        call_fn: CallFn,
        *,
        group: str = "custom",
    ) -> None:
        """Register a raw callable for progressive disclosure."""
        self._register_tool(
            _RegisteredTool(
                name=name,
                description=description,
                schema=schema,
                source_id=group,
                call_fn=call_fn,
            )
        )

    def _register_meta_tools(self) -> None:
        if "list_tools" not in self._exclude_meta_tools:

            @self.tool
            async def list_tools(ctx: RunContext[Any]) -> dict[str, str]:
                return await self._list_tools_impl(ctx)

        if "search_tools" not in self._exclude_meta_tools:

            @self.tool
            async def search_tools(ctx: RunContext[Any], query: str) -> dict[str, str]:
                return await self._search_tools_impl(ctx, query)

        if "get_tool_schema" not in self._exclude_meta_tools:

            @self.tool
            async def get_tool_schema(ctx: RunContext[Any], tool_name: str) -> str:
                return await self._get_tool_schema_impl(ctx, tool_name)

        if "call_tool" not in self._exclude_meta_tools:

            @self.tool
            async def call_tool(
                ctx: RunContext[Any],
                tool_name: str,
                arguments: dict[str, Any] | None = None,
            ) -> Any:
                return await self._call_tool_impl(ctx, tool_name, arguments)

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
        q = query.lower()
        return {
            name: tool.description[:200]
            for name, tool in sorted(self._registry.items())
            if q in name.lower() or q in tool.description.lower()
        }

    async def _get_tool_schema_impl(self, ctx: RunContext[Any], tool_name: str) -> str:
        await self._resolve_pending(ctx)
        tool = self._registry.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._registry)[:20])
            raise ValueError(
                f"Tool '{tool_name}' not found. Some available tools: {available}"
            )

        return json.dumps(
            {
                "name": tool.name,
                "description": tool.description,
                "group": tool.source_id,
                "parameters": tool.schema,
            },
            indent=2,
        )

    async def _call_tool_impl(
        self,
        ctx: RunContext[Any],
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        await self._resolve_pending(ctx)
        tool = self._registry.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._registry)[:20])
            raise ValueError(
                f"Tool '{tool_name}' not found. Some available tools: {available}"
            )

        args = arguments or {}
        if tool.source_toolset is not None and tool.tool_handle is not None:
            return await tool.source_toolset.call_tool(
                tool_name,
                args,
                ctx,
                tool.tool_handle,
            )

        if tool.call_fn is None:
            raise RuntimeError(
                f"Tool '{tool_name}' has no executable callable configured."
            )

        result = tool.call_fn(ctx, tool_name, args)
        if inspect.isawaitable(result):
            return await result
        return result

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
        return self._instruction_template.format(tool_groups=tool_groups)

    @property
    def discovered_tools(self) -> dict[str, str]:
        """Snapshot of currently discovered tool names and descriptions."""
        return {name: tool.description for name, tool in sorted(self._registry.items())}


def progressive_toolset(
    *toolsets: AbstractToolset[Any],
    group_descriptions: dict[str, str] | None = None,
    **kwargs: Any,
) -> ToolDiscoveryToolset:
    """Shorthand helper for wrapping multiple toolsets."""
    return ToolDiscoveryToolset(
        toolsets=list(toolsets),
        group_descriptions=group_descriptions,
        **kwargs,
    )
