from __future__ import annotations

from typing import Callable, Sequence
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
