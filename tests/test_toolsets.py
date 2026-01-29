from __future__ import annotations

from unittest.mock import MagicMock

from openbb_ai.models import MessageArtifactSSE

from openbb_pydantic_ai._viz_toolsets import (
    HtmlParams,
    _create_html,
    _html_artifact,
)
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name


def test_html_artifact_returns_message_artifact_sse() -> None:
    artifact = _html_artifact(
        content="<div>Hello World</div>",
        name="Test HTML",
        description="A test HTML artifact",
    )

    assert isinstance(artifact, MessageArtifactSSE)
    assert artifact.data.type == "html"
    assert artifact.data.name == "Test HTML"
    assert artifact.data.description == "A test HTML artifact"
    assert artifact.data.content == "<div>Hello World</div>"


def test_html_artifact_uses_defaults_when_optional_params_missing() -> None:
    artifact = _html_artifact(content="<p>Content</p>")

    assert artifact.data.name == "HTML Content"
    assert artifact.data.description == "HTML artifact"
    assert artifact.data.content == "<p>Content</p>"


def test_html_params_validates_content() -> None:
    params = HtmlParams(
        content="<span>Test</span>",
        name="My HTML",
        description="A description",
    )

    assert params.content == "<span>Test</span>"
    assert params.name == "My HTML"
    assert params.description == "A description"


def test_html_params_allows_optional_fields_to_be_none() -> None:
    params = HtmlParams(content="<div>Only content</div>")

    assert params.content == "<div>Only content</div>"
    assert params.name is None
    assert params.description is None


def test_create_html_returns_tool_return_with_artifact() -> None:
    ctx = MagicMock()
    params = HtmlParams(
        content="<article>Article content</article>",
        name="Article",
        description="An article artifact",
    )

    result = _create_html(ctx, params)

    assert result.return_value == "HTML artifact created successfully."
    assert "html" in result.metadata
    html_artifact = result.metadata["html"]
    assert isinstance(html_artifact, MessageArtifactSSE)
    assert html_artifact.data.type == "html"
    assert html_artifact.data.content == "<article>Article content</article>"
    assert html_artifact.data.name == "Article"
    assert html_artifact.data.description == "An article artifact"


def test_build_widget_tool_name_ignores_origin(widget_with_origin) -> None:
    widget = widget_with_origin("OpenBB Sandbox")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"


def test_build_widget_tool_name_ignores_partner_origin(widget_with_origin) -> None:
    widget = widget_with_origin("Partner API")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"


def test_build_widget_tool_name_handles_plain_openbb_origin(widget_with_origin) -> None:
    widget = widget_with_origin("OpenBB")

    assert build_widget_tool_name(widget) == "openbb_widget_financial_statements"
