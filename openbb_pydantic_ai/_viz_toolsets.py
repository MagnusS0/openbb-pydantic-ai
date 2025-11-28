from __future__ import annotations

from typing import Any, Literal

from openbb_ai.helpers import chart, table
from pydantic import BaseModel, model_validator
from pydantic_ai import FunctionToolset, ToolReturn
from pydantic_ai.tools import RunContext

from openbb_pydantic_ai._dependencies import OpenBBDeps


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


def _create_table(
    ctx: RunContext[OpenBBDeps],
    data: list[dict[Any, Any]],
    name: str | None = None,
    description: str | None = None,
) -> ToolReturn:
    """
    Create a table artifact.

    Always requires:
    - data: List of data points (dictionaries)
    """
    return ToolReturn(
        return_value="Table created successfully.",
        metadata={
            "table": table(
                data=data,
                name=name,
                description=description,
            )
        },
    )


def build_viz_toolsets() -> FunctionToolset[OpenBBDeps]:
    """Create visualization toolsets including chart and table creation tools.

    Returns
    -------
    tuple[FunctionToolset[OpenBBDeps], ...]
        Toolsets including visualization tools
    """
    viz_toolset = FunctionToolset[OpenBBDeps]()
    viz_toolset.add_function(_create_chart, name="openbb_create_chart")
    viz_toolset.add_function(_create_table, name="openbb_create_table")

    return viz_toolset
