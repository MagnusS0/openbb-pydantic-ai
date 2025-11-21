"""Toolset implementations for OpenBB widgets and visualization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

from openbb_ai.helpers import chart
from openbb_ai.models import Undefined, Widget, WidgetCollection, WidgetParam
from pydantic import BaseModel, model_validator
from pydantic_ai import CallDeferred, Tool, ToolReturn
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import FunctionToolset

from openbb_pydantic_ai._config import PARAM_TYPE_SCHEMA_MAP
from openbb_pydantic_ai._dependencies import OpenBBDeps


def _base_param_schema(param: WidgetParam) -> dict[str, Any]:
    """Build the base JSON schema for a widget parameter."""
    schema = PARAM_TYPE_SCHEMA_MAP.get(param.type, {"type": "string"})
    schema = dict(schema)  # copy
    schema["description"] = param.description

    if param.options:
        schema["enum"] = list(param.options)

    if param.get_options:
        schema.setdefault(
            "description",
            param.description + " (options retrieved dynamically)",
        )

    if param.default_value is not Undefined.UNDEFINED:
        schema["default"] = param.default_value

    if param.current_value is not None and param.multi_select is False:
        schema.setdefault("examples", []).append(param.current_value)

    return schema


def _param_schema(param: WidgetParam) -> tuple[dict[str, Any], bool]:
    """Return the schema for a parameter and whether it's required."""
    schema = _base_param_schema(param)

    if param.multi_select:
        schema = {
            "type": "array",
            "items": schema,
            "description": schema.get("description"),
        }

    is_required = param.default_value is Undefined.UNDEFINED
    return schema, is_required


def _widget_schema(widget: Widget) -> dict[str, Any]:
    """Build the JSON schema for a widget's parameters."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param in widget.params:
        schema, is_required = _param_schema(param)
        properties[param.name] = schema
        if is_required:
            required.append(param.name)

    widget_schema: dict[str, Any] = {
        "type": "object",
        "title": widget.name,
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        widget_schema["required"] = required

    return widget_schema


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return slug.lower() or "value"


def build_widget_tool_name(widget: Widget) -> str:
    """Generate a deterministic (and reasonably short) widget tool name.

    Names always start with the ``openbb_widget`` prefix so Workspace can route
    them, followed by the widget identifier and, when helpful, a trimmed origin
    slug. When the origin already starts with ``openbb_`` (e.g. ``OpenBB Sandbox``)
    the redundant ``openbb`` portion is removed to avoid unwieldy names like
    ``openbb_widget_openbb_sandbox_*``.
    """

    origin_slug = _slugify(widget.origin)
    if origin_slug.startswith("openbb_"):
        origin_slug = origin_slug.removeprefix("openbb_")
    elif origin_slug == "openbb":
        origin_slug = ""

    widget_slug = _slugify(widget.widget_id)

    parts = ["openbb", "widget"]
    if origin_slug:
        parts.append(origin_slug)
    parts.append(widget_slug)
    return "_".join(parts)


def build_widget_tool(widget: Widget) -> Tool:
    """Create a deferred tool for a widget.

    This creates a Pydantic AI tool that will be called by the LLM but
    executed by the OpenBB Workspace frontend (deferred execution).

    Parameters
    ----------
    widget : Widget
        The widget to create a tool for

    Returns
    -------
    Tool
        A Tool configured for deferred execution
    """
    tool_name = build_widget_tool_name(widget)
    schema = _widget_schema(widget)
    description = widget.description or widget.name

    async def _call_widget(ctx: RunContext[OpenBBDeps], **input_arguments: Any) -> None:
        # Ensure we have a tool call id for deferred execution
        if ctx.tool_call_id is None:
            raise RuntimeError("Deferred widget tools require a tool call id.")
        raise CallDeferred

    _call_widget.__name__ = f"call_widget_{widget.uuid}"

    return Tool.from_schema(
        function=_call_widget,
        name=tool_name,
        description=description,
        json_schema=schema,
        takes_ctx=True,
    )


class ChartParams(BaseModel):
    """Validation model for chart creation parameters."""

    type: Literal["line", "bar", "scatter", "pie", "donut"]
    data: list[dict[str, Any]]
    x_key: str | None = None
    y_keys: list[str] | None = None
    angle_key: str | None = None
    callout_label_key: str | None = None
    name: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_chart_keys(self) -> ChartParams:
        if self.type in {"line", "bar", "scatter"}:
            if not self.x_key:
                raise ValueError("x_key is required for line, bar, and scatter charts")
            if not self.y_keys:
                raise ValueError("y_keys is required for line, bar, and scatter charts")
        elif self.type in {"pie", "donut"}:
            if not self.angle_key:
                raise ValueError("angle_key is required for pie and donut charts")
            if not self.callout_label_key:
                raise ValueError(
                    "callout_label_key is required for pie and donut charts"
                )
        return self


def _create_chart(ctx: RunContext[OpenBBDeps], params: ChartParams) -> ToolReturn:
    """
    Create a chart artifact (line, bar, scatter, pie, donut).

    Required params for line, bar, scatter charts: x_key, y_keys.
    Required params for pie, donut charts: angle_key, callout_label_key.

    Always requires:
    - type: Chart type (line, bar, scatter, pie, donut)
    - data: List of data points (dictionaries)
    """
    return ToolReturn(
        return_value="Chart created successfully.",
        metadata={
            "chart": chart(
                type=params.type,
                data=params.data,
                x_key=params.x_key,
                y_keys=params.y_keys,
                angle_key=params.angle_key,
                callout_label_key=params.callout_label_key,
                name=params.name,
                description=params.description,
            )
        },
    )


class WidgetToolset(FunctionToolset[OpenBBDeps]):
    """Toolset that exposes widgets as deferred tools."""

    def __init__(self, widgets: Sequence[Widget]):
        super().__init__()
        self._widgets_by_tool: dict[str, Widget] = {}

        for widget in widgets:
            tool = build_widget_tool(widget)
            self.add_tool(tool)
            self._widgets_by_tool[tool.name] = widget

        self.widgets_by_tool: Mapping[str, Widget] = MappingProxyType(
            self._widgets_by_tool
        )


def build_widget_toolsets(
    collection: WidgetCollection | None,
) -> tuple[FunctionToolset[OpenBBDeps], ...]:
    """Create toolsets for each widget priority group plus visualization tools.

    Widgets are organized into separate toolsets by priority (primary, secondary, extra)
    to allow control over tool selection. The visualization toolset is always
    included for creating charts and tables.

    Parameters
    ----------
    collection : WidgetCollection | None
        Widget collection with priority groups, or None

    Returns
    -------
    tuple[FunctionToolset[OpenBBDeps], ...]
        Toolsets including widget toolsets and visualization toolset
    """
    viz_toolset = FunctionToolset[OpenBBDeps]()
    viz_toolset.add_function(_create_chart, name="openbb_create_chart")

    if collection is None:
        return (viz_toolset,)

    toolsets: list[FunctionToolset[OpenBBDeps]] = []
    for widgets in (collection.primary, collection.secondary, collection.extra):
        if widgets:
            toolsets.append(WidgetToolset(widgets))

    toolsets.append(viz_toolset)

    return tuple(toolsets)
