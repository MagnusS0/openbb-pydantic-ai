from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    DashboardInfo,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    QueryRequest,
    RoleEnum,
    WorkspaceState,
)

from openbb_pydantic_ai import OpenBBAIAdapter
from openbb_pydantic_ai._config import (
    CONTEXT_SNAPSHOT_EXTRA_STATE_KEY,
    EXECUTE_MCP_TOOL_NAME,
)
from openbb_pydantic_ai._dependencies import build_context_snapshot
from tests.unit.adapter._assertions import tool_return_parts

pytestmark = pytest.mark.regression_contract

_DASHBOARD = DashboardInfo(id="d1", name="Macro", current_tab_id="tab-1", tabs=None)


def _snapshot() -> dict[str, Any]:
    original = QueryRequest(
        messages=[LlmClientMessage(role=RoleEnum.human, content="Hi")],
        timezone="Europe/Oslo",
        urls=["https://example.com"],
        workspace_state=WorkspaceState(current_dashboard_info=_DASHBOARD),
    )
    return build_context_snapshot(original)


def _call_and_result(snapshot: dict[str, Any] | None):
    args = {"server_id": "s", "tool_name": "mcp_a", "parameters": {}}
    extra_state: dict[str, Any] = {
        "tool_calls": [{"tool_call_id": "m-1", "tool_name": "mcp_a"}]
    }
    if snapshot is not None:
        extra_state[CONTEXT_SNAPSHOT_EXTRA_STATE_KEY] = snapshot
    return [
        LlmClientMessage(
            role=RoleEnum.ai,
            content=LlmClientFunctionCall(
                function=EXECUTE_MCP_TOOL_NAME, input_arguments=args
            ),
        ),
        LlmClientFunctionCallResultMessage(
            function=EXECUTE_MCP_TOOL_NAME,
            input_arguments=args,
            data=[ClientCommandResult(status="success", message=None)],
            extra_state=extra_state,
        ),
    ]


def _adapter(request: QueryRequest) -> OpenBBAIAdapter:
    return OpenBBAIAdapter(agent=MagicMock(), run_input=request)


def _follow_up(**kwargs: Any) -> QueryRequest:
    human = LlmClientMessage(role=RoleEnum.human, content="Hi")
    return QueryRequest(
        messages=[human, *_call_and_result(_snapshot())],
        **kwargs,
    )


def test_follow_up_restores_missing_context_from_trailing_result() -> None:
    adapter = _adapter(_follow_up())

    assert adapter.deps.timezone == "Europe/Oslo"
    assert adapter.deps.urls == ["https://example.com"]
    assert adapter.deps.workspace_state is not None
    assert adapter.deps.workspace_state.current_dashboard_info == _DASHBOARD
    assert adapter.deps.state["current_dashboard_info"]["name"] == "Macro"
    assert "tab-1" in adapter.instructions
    assert "Europe/Oslo" in adapter.instructions


def test_request_values_win_over_snapshot() -> None:
    other = DashboardInfo(id="d2", name="Other", current_tab_id="t", tabs=None)
    adapter = _adapter(
        _follow_up(
            timezone="Asia/Tokyo",
            urls=["https://request.example"],
            workspace_state=WorkspaceState(current_dashboard_info=other),
        )
    )

    assert adapter.deps.timezone == "Asia/Tokyo"
    assert adapter.deps.urls == ["https://request.example"]
    assert adapter.deps.workspace_state is not None
    assert adapter.deps.workspace_state.current_dashboard_info == other


def test_snapshot_from_an_earlier_chain_is_not_restored() -> None:
    messages = [
        LlmClientMessage(role=RoleEnum.human, content="first question"),
        *_call_and_result(_snapshot()),
        LlmClientMessage(role=RoleEnum.ai, content="answer"),
        LlmClientMessage(role=RoleEnum.human, content="second question"),
    ]

    adapter = _adapter(QueryRequest(messages=messages))

    assert adapter.deps.timezone == "UTC"
    assert adapter.deps.urls is None
    assert adapter.deps.workspace_state is None


def test_late_result_after_assistant_text_does_not_restore() -> None:
    messages = [
        LlmClientMessage(role=RoleEnum.human, content="question"),
        LlmClientMessage(role=RoleEnum.ai, content="answer"),
        *_call_and_result(_snapshot())[1:],
    ]

    adapter = _adapter(QueryRequest(messages=messages))

    assert adapter.deps.timezone == "UTC"
    assert adapter.deps.urls is None


def test_snapshot_never_reaches_model_visible_history() -> None:
    adapter = _adapter(_follow_up())

    assert tool_return_parts(adapter)
    history = str(adapter.messages)
    assert CONTEXT_SNAPSHOT_EXTRA_STATE_KEY not in history
    assert "Europe/Oslo" not in history
