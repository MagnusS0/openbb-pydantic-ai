from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    DashboardInfo,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
    Widget,
    WidgetCollection,
    WidgetParam,
    WorkspaceState,
)
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from openbb_pydantic_ai import OpenBBAIAdapter
from openbb_pydantic_ai._widget_toolsets import build_widget_tool_name


def test_adapter_injects_instructions(sample_context, make_request, human_message):
    request = make_request(
        [human_message],
        context=[sample_context],
        urls=["https://example.com"],
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    instructions = adapter.instructions

    assert instructions, "Adapter should inject context instructions"
    assert "Test Context" in instructions
    assert "https://example.com" in instructions


@pytest.mark.anyio
async def test_from_request_preprocesses_messages(mocker, make_request) -> None:
    """Adapter should preprocess PDF-bearing messages before transform/build."""
    from openbb_pydantic_ai import _adapter as adapter_module

    original_message = LlmClientMessage(role=RoleEnum.human, content="original")
    processed_message = LlmClientMessage(role=RoleEnum.human, content="processed")
    run_input = make_request([original_message])

    preprocess_mock = mocker.patch.object(
        adapter_module,
        "preprocess_pdf_in_messages",
        new_callable=AsyncMock,
        return_value=[processed_message],
    )

    class _RequestStub:
        headers = {"accept": "text/event-stream"}

        def __init__(self, payload: bytes):
            self._payload = payload

        async def body(self) -> bytes:
            return self._payload

    request = _RequestStub(run_input.model_dump_json().encode())
    adapter = await OpenBBAIAdapter.from_request(request, agent=MagicMock())

    preprocess_mock.assert_awaited_once()
    assert adapter._base_messages == [processed_message]


def test_adapter_preserves_tool_call_ids(widget_collection, make_request):
    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=tool_name,
            input_arguments={"symbol": "AAPL"},
        ),
    )
    result_message = LlmClientFunctionCallResultMessage(
        function=tool_name,
        input_arguments={"symbol": "AAPL"},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={"tool_calls": [{"tool_call_id": "tool-123"}]},
    )
    request = make_request(
        [call_message, result_message],
        widgets=widget_collection,
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    tool_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]
    assert tool_parts and tool_parts[0].tool_call_id == "tool-123"

    return_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]
    assert return_parts and return_parts[0].tool_call_id == "tool-123"

    # Trailing tool results are kept for SSE replay
    assert adapter._pending_results == [result_message]


def test_adapter_uses_extra_state_tool_calls_id(widget_collection, make_request):
    """Deferred results must retain the exact tool_call_id emitted earlier.

    Regression: during the refactor we only looked at ``extra_state['tool_call_id']``
    and ignored the ``extra_state['tool_calls'][0]['tool_call_id']`` that is set when
    get_widget_data/MCP calls are dispatched. That mismatch caused prior tool
    responses to disappear on the next turn because their IDs were re-hashed.
    """

    widget = widget_collection.primary[0]
    tool_name = build_widget_tool_name(widget)

    call_message = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function=tool_name,
            input_arguments={"symbol": "NVDA"},
        ),
    )

    # Simulate how the event stream encodes the tool id inside extra_state.tool_calls
    tool_call_id = "call-456"
    result_message = LlmClientFunctionCallResultMessage(
        function=tool_name,
        input_arguments={"symbol": "NVDA"},
        data=[ClientCommandResult(status="success", message=None)],
        extra_state={
            "tool_calls": [
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                }
            ]
        },
    )

    request = make_request([call_message, result_message], widgets=widget_collection)

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    # The transformed history should carry through the original tool_call_id
    parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ]

    assert {p.tool_call_id for p in parts} == {tool_call_id}

    # Pending results are preserved for SSE replay rather than deferred resends
    assert adapter._pending_results == [result_message]


def test_adapter_unbatches_multiple_widget_calls(widget_collection, make_request):
    """Batched widget calls should be unbatched on both sides for proper matching.

    When the UI returns a batched get_widget_data call with multiple data_sources,
    both the call and result should be unbatched so each ToolCallPart has a matching
    ToolReturnPart with the same tool_call_id.
    """
    widget = widget_collection.primary[0]

    # The adapter batches widget calls into get_widget_data
    # So the call messages in history will be get_widget_data, not individual widgets
    batched_call = LlmClientMessage(
        role=RoleEnum.ai,
        content=LlmClientFunctionCall(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {
                        "widget_uuid": str(widget.uuid),
                        "origin": widget.origin,
                        "id": widget.widget_id,
                        "input_args": {
                            "symbol": "AAPL",
                            "selectedGroup": "balance_sheet",
                        },
                    },
                    {
                        "widget_uuid": str(widget.uuid),
                        "origin": widget.origin,
                        "id": widget.widget_id,
                        "input_args": {
                            "symbol": "AAPL",
                            "selectedGroup": "income_statement",
                        },
                    },
                ]
            },
        ),
    )

    # Batched result from get_widget_data with both tool_call_ids
    batched_result = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={
            "data_sources": [
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {"symbol": "AAPL", "selectedGroup": "balance_sheet"},
                },
                {
                    "widget_uuid": str(widget.uuid),
                    "origin": widget.origin,
                    "id": widget.widget_id,
                    "input_args": {
                        "symbol": "AAPL",
                        "selectedGroup": "income_statement",
                    },
                },
            ]
        },
        data=[
            ClientCommandResult(status="success", message="Balance sheet data"),
            ClientCommandResult(status="success", message="Income statement data"),
        ],
        extra_state={
            "tool_calls": [
                {"tool_call_id": "call_123", "widget_uuid": str(widget.uuid)},
                {"tool_call_id": "call_456", "widget_uuid": str(widget.uuid)},
            ]
        },
    )

    request = make_request(
        [batched_call, batched_result],
        widgets=widget_collection,
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    # Both the call and result should be unbatched into 2 parts each
    tool_call_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]
    tool_return_parts = [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]

    assert len(tool_call_parts) == 2  # Unbatched into 2 calls
    assert len(tool_return_parts) == 2  # Unbatched into 2 returns

    # Each call should match a return via tool_call_id
    call_ids = {p.tool_call_id for p in tool_call_parts}
    return_ids = {p.tool_call_id for p in tool_return_parts}
    assert call_ids == return_ids == {"call_123", "call_456"}

    # Each call should have a single data_source (unbatched)
    for part in tool_call_parts:
        data_sources = (
            part.args.get("data_sources", []) if isinstance(part.args, dict) else []
        )
        assert len(data_sources) == 1


def test_adapter_preserves_turn_boundaries_without_duplication(make_request):
    """Ensure messages are not re-grouped or duplicated across turns."""

    first_user = LlmClientMessage(
        role=RoleEnum.human, content="Hey can you get ticker info on AAPL"
    )
    assistant_reply = LlmClientMessage(
        role=RoleEnum.ai,
        content="Here's the latest ticker information for AAPL.",
    )
    follow_up = LlmClientMessage(
        role=RoleEnum.human,
        content="How many times have you shown that ticker info?",
    )

    request = make_request([first_user, assistant_reply, follow_up])

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    messages = adapter.messages

    # Extract visible text content per turn to verify ordering and isolation
    turn_text = []
    for msg in messages:
        parts_text = [
            part.content
            for part in msg.parts
            if isinstance(part, (UserPromptPart, TextPart))
        ]
        if parts_text:
            turn_text.append(" ".join(str(text) for text in parts_text))

    assert len(turn_text) == 3
    assert turn_text[0] == "Hey can you get ticker info on AAPL"
    assert turn_text[1] == "Here's the latest ticker information for AAPL."
    assert turn_text[2] == "How many times have you shown that ticker info?"


@pytest.mark.anyio
async def test_run_stream_native_does_not_double_messages(
    make_request, agent_stream_stub
):
    """Base UIAdapter appends self.messages; ensure we don't pass them twice."""

    user_msg = LlmClientMessage(role=RoleEnum.human, content="Hi")
    request = make_request([user_msg])

    adapter = OpenBBAIAdapter(agent=agent_stream_stub, run_input=request)

    stream = adapter.run_stream_native()
    async for _ in stream:
        pass

    assert len(agent_stream_stub.calls) == 1
    _, call_kwargs = agent_stream_stub.calls[0]
    history = call_kwargs["message_history"]

    # Should match adapter.messages exactly (not doubled)
    assert history == adapter.messages
    assert len(history) == 1


@pytest.mark.anyio
async def test_run_stream_does_not_double_messages(make_request, agent_stream_stub):
    """Same guarantee for the higher-level run_stream wrapper."""

    user_msg = LlmClientMessage(role=RoleEnum.human, content="Hello again")
    request = make_request([user_msg])

    adapter = OpenBBAIAdapter(agent=agent_stream_stub, run_input=request)

    stream = adapter.run_stream()
    async for _ in stream:
        pass

    assert len(agent_stream_stub.calls) == 1
    _, call_kwargs = agent_stream_stub.calls[0]
    history = call_kwargs["message_history"]

    assert history == adapter.messages
    assert len(history) == 1


def test_adapter_dashboard_info_formatting(make_request, human_message):
    # Create widgets with params
    widget1 = Widget(
        origin="test",
        widget_id="widget1",
        name="Widget One",
        description="Desc 1",
        params=[
            WidgetParam(
                name="symbol", type="text", description="desc", current_value="AAPL"
            ),
            WidgetParam(
                name="period", type="text", description="desc", default_value="1y"
            ),
        ],
    )
    widget2 = Widget(
        origin="test",
        widget_id="widget2",
        name="Widget Two",
        description="Desc 2",
        params=[],
    )

    widgets = WidgetCollection(primary=[widget1, widget2])

    # Create dashboard info using dict and model_validate
    dashboard_data = {
        "id": str(uuid4()),
        "name": "Test Dashboard",
        "current_tab_id": "tab1",
        "tabs": [
            {
                "tab_id": "tab1",
                "widgets": [
                    {"widget_uuid": str(widget1.uuid), "name": "Widget One Custom"}
                ],
            },
            {
                "tab_id": "tab2",
                "widgets": [{"widget_uuid": str(widget2.uuid), "name": "Widget Two"}],
            },
        ],
    }

    workspace_state = WorkspaceState(
        current_dashboard_info=DashboardInfo.model_validate(dashboard_data)
    )

    request = make_request(
        [human_message],
        widgets=widgets,
        workspace_state=workspace_state,
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

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
    # Widget 1 in dashboard, Widget 2 not in dashboard
    widget1 = Widget(
        origin="test", widget_id="w1", name="W1", description="d", params=[]
    )
    widget2 = Widget(
        origin="test", widget_id="w2", name="W2", description="d", params=[]
    )

    widgets = WidgetCollection(primary=[widget1, widget2])

    dashboard_data = {
        "id": str(uuid4()),
        "name": "Dash",
        "current_tab_id": "t1",
        "tabs": [
            {
                "tab_id": "t1",
                "widgets": [{"widget_uuid": str(widget1.uuid), "name": "W1"}],
            }
        ],
    }

    request = make_request(
        [human_message],
        widgets=widgets,
        workspace_state=WorkspaceState(
            current_dashboard_info=DashboardInfo.model_validate(dashboard_data)
        ),
    )

    adapter = OpenBBAIAdapter(agent=MagicMock(), run_input=request)

    instructions = adapter.instructions
    content = instructions

    assert "Widgets by Tab:" in content
    assert "## t1" in content
    assert "- W1" in content

    assert "Other Available Widgets:" in content
    assert "- W2" in content
