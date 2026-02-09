from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import pytest
from openbb_ai.models import (
    DataContent,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    QueryRequest,
    RawContext,
    RoleEnum,
    SingleDataContent,
    Widget,
    WidgetCollection,
)


@pytest.fixture
def sample_widget() -> Widget:
    return Widget(
        origin="OpenBB API",
        widget_id="sample_widget",
        name="Sample Widget",
        description="Widget used for testing.",
        params=[],
        metadata={},
    )


@pytest.fixture
def widget_collection(sample_widget: Widget) -> WidgetCollection:
    return WidgetCollection(primary=[sample_widget])


@pytest.fixture
def sample_context() -> RawContext:
    return RawContext(
        uuid=uuid4(),
        name="Test Context",
        description="Context description",
        data=DataContent(items=[SingleDataContent(content="{}")]),
    )


@pytest.fixture
def make_request() -> Callable[..., QueryRequest]:
    def _factory(
        messages: Sequence[LlmClientMessage | LlmClientFunctionCallResultMessage],
        **kwargs,
    ) -> QueryRequest:
        return QueryRequest(messages=list(messages), **kwargs)

    return _factory


@pytest.fixture
def human_message() -> LlmClientMessage:
    return LlmClientMessage(role=RoleEnum.human, content="Hello")


class AgentStreamStub:
    """Lightweight stub that mimics Agent.run_stream_events for unit tests."""

    def __init__(
        self,
        stream_fn: Callable[..., AsyncIterator[Any]],
        output_type: Any | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.output_type = output_type if output_type is not None else object()

    def run_stream_events(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append((args, kwargs))
        return self._stream_fn(*args, **kwargs)


@pytest.fixture
def agent_stream_stub() -> AgentStreamStub:
    async def _empty_stream(*, message_history=None, **_: Any) -> AsyncIterator[Any]:
        assert message_history is not None
        if False:
            yield

    return AgentStreamStub(_empty_stream)


@pytest.fixture
def widget_with_origin(sample_widget: Widget) -> Callable[[str, str], Widget]:
    def _factory(origin: str, widget_id: str = "financial_statements") -> Widget:
        if hasattr(sample_widget, "model_copy"):
            return sample_widget.model_copy(
                update={"origin": origin, "widget_id": widget_id}
            )
        clone = Widget(**sample_widget.__dict__)
        clone.origin = origin
        clone.widget_id = widget_id
        return clone

    return _factory


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


_PDF_TEST_PATH = Path(__file__).parent / "fixtures" / "PDF_TestPage.pdf"


@pytest.fixture
def sample_pdf_base64() -> str:
    """Load the test PDF as a base64-encoded string."""
    return base64.b64encode(_PDF_TEST_PATH.read_bytes()).decode()


@pytest.fixture
def pdf_data_format() -> dict[str, str]:
    """PDF data format dict for constructing PdfDataFormat in tests."""
    return {"data_type": "pdf", "filename": "test.pdf"}
