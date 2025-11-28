from __future__ import annotations

from openbb_ai.models import LlmClientMessage, QueryRequest, RawContext, RoleEnum

from openbb_pydantic_ai._dependencies import build_deps_from_request


def test_build_deps_from_request(sample_context: RawContext) -> None:
    request = QueryRequest(
        messages=[LlmClientMessage(role=RoleEnum.human, content="Hello")],
        context=[sample_context],
        urls=["https://example.com"],
        timezone="America/New_York",
    )

    deps = build_deps_from_request(request)

    assert deps.timezone == "America/New_York"
    assert deps.urls == ["https://example.com"]
    assert deps.context and deps.context[0].name == "Test Context"
    assert deps.state == {}
