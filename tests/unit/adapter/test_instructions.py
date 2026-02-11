from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from openbb_ai.models import (
    DashboardInfo,
    Widget,
    WidgetCollection,
    WidgetParam,
    WorkspaceState,
)

from openbb_pydantic_ai import OpenBBAIAdapter

pytestmark = pytest.mark.regression_contract


def _make_widget(
    *, widget_id: str, name: str, params: list[WidgetParam] | None = None
) -> Widget:
    return Widget(
        origin="test",
        widget_id=widget_id,
        name=name,
        description=f"{name} description",
        params=params or [],
    )


def _make_workspace_state(
    *,
    dashboard_name: str,
    current_tab_id: str,
    tabs: list[tuple[str, list[tuple[Widget, str | None]]]],
) -> WorkspaceState:
    tab_entries: list[dict[str, object]] = []
    for tab_id, tab_widgets in tabs:
        tab_entries.append(
            {
                "tab_id": tab_id,
                "widgets": [
                    {
                        "widget_uuid": str(widget.uuid),
                        "name": widget_name if widget_name is not None else widget.name,
                    }
                    for widget, widget_name in tab_widgets
                ],
            }
        )

    dashboard_data = {
        "id": str(uuid4()),
        "name": dashboard_name,
        "current_tab_id": current_tab_id,
        "tabs": tab_entries,
    }
    return WorkspaceState(
        current_dashboard_info=DashboardInfo.model_validate(dashboard_data)
    )


def _build_adapter(make_request, human_message, **kwargs) -> OpenBBAIAdapter:
    request = make_request([human_message], **kwargs)
    return OpenBBAIAdapter(agent=MagicMock(), run_input=request)


def test_adapter_injects_instructions(sample_context, make_request, human_message):
    adapter = _build_adapter(
        make_request,
        human_message,
        context=[sample_context],
        urls=["https://example.com"],
    )

    instructions = adapter.instructions

    assert instructions, "Adapter should inject context instructions"
    assert "Test Context" in instructions
    assert "https://example.com" in instructions


def test_adapter_dashboard_info_formatting(make_request, human_message):
    widget1 = _make_widget(
        widget_id="widget1",
        name="Widget One",
        params=[
            WidgetParam(
                name="symbol", type="text", description="desc", current_value="AAPL"
            ),
            WidgetParam(
                name="period", type="text", description="desc", default_value="1y"
            ),
        ],
    )
    widget2 = _make_widget(widget_id="widget2", name="Widget Two")

    widgets = WidgetCollection(primary=[widget1, widget2])
    workspace_state = _make_workspace_state(
        dashboard_name="Test Dashboard",
        current_tab_id="tab1",
        tabs=[
            ("tab1", [(widget1, "Widget One Custom")]),
            ("tab2", [(widget2, "Widget Two")]),
        ],
    )

    adapter = _build_adapter(
        make_request,
        human_message,
        widgets=widgets,
        workspace_state=workspace_state,
    )

    instructions = adapter.instructions

    assert instructions
    content = instructions

    assert "<dashboard_info>" in content
    assert "Active dashboard: Test Dashboard" in content
    assert "Current tab: tab1" in content

    assert "Widgets by Tab:" in content
    assert "## tab1" in content
    # Check for custom name and params
    assert "- Widget One Custom: symbol=AAPL, period=1y (default)" in content

    assert "## tab2" in content
    assert "- Widget Two" in content

    assert "</dashboard_info>" in content

    # Ensure no duplicate widget defaults section
    assert "<widget_defaults>" not in content


def test_adapter_dashboard_orphaned_widgets(make_request, human_message):
    widget1 = _make_widget(widget_id="w1", name="W1")
    widget2 = _make_widget(widget_id="w2", name="W2")

    widgets = WidgetCollection(primary=[widget1, widget2])
    workspace_state = _make_workspace_state(
        dashboard_name="Dash",
        current_tab_id="t1",
        tabs=[("t1", [(widget1, "W1")])],
    )

    adapter = _build_adapter(
        make_request,
        human_message,
        widgets=widgets,
        workspace_state=workspace_state,
    )

    instructions = adapter.instructions
    content = instructions

    assert "Widgets by Tab:" in content
    assert "## t1" in content
    assert "- W1" in content

    assert "Other Available Widgets:" in content
    assert "- W2" in content
