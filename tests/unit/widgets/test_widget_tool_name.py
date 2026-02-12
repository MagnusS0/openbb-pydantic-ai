from __future__ import annotations

from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name


def test_build_widget_tool_name_ignores_origin(widget_with_origin) -> None:
    widget = widget_with_origin("OpenBB Sandbox")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"


def test_build_widget_tool_name_ignores_partner_origin(widget_with_origin) -> None:
    widget = widget_with_origin("Partner API")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"


def test_build_widget_tool_name_handles_plain_openbb_origin(widget_with_origin) -> None:
    widget = widget_with_origin("OpenBB")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"
