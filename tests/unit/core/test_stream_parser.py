from __future__ import annotations

from collections.abc import Iterator

from openbb_ai.helpers import chart
from openbb_ai.models import MessageArtifactSSE, MessageChunkSSE

from openbb_pydantic_ai._config import CHART_PLACEHOLDER_TOKEN
from openbb_pydantic_ai._stream_parser import StreamParser


def _artifact_iter(artifact: MessageArtifactSSE) -> Iterator[MessageArtifactSSE]:
    yield artifact


def _sample_chart() -> MessageArtifactSSE:
    return chart(
        type="bar",
        data=[{"label": "A", "value": 1}],
        x_key="label",
        y_keys=["value"],
    )


def test_stream_parser_swaps_canonical_placeholder_token() -> None:
    parser = StreamParser()
    artifact = _sample_chart()

    events = parser.parse(
        f"Intro {CHART_PLACEHOLDER_TOKEN} Outro",
        _artifact_iter(artifact),
    )

    assert len(events) == 3
    assert isinstance(events[0], MessageChunkSSE)
    assert events[0].data.delta == "Intro "
    assert events[1] is artifact
    assert isinstance(events[2], MessageChunkSSE)
    assert events[2].data.delta == " Outro"


def test_stream_parser_buffers_until_artifact_available() -> None:
    parser = StreamParser()
    artifact = _sample_chart()

    events = parser.parse(
        f"Start {CHART_PLACEHOLDER_TOKEN} After",
        iter(()),
    )

    assert len(events) == 1
    assert isinstance(events[0], MessageChunkSSE)
    assert events[0].data.delta == "Start "
    assert parser.has_pending_placeholder()

    follow_up_events = parser.parse("", _artifact_iter(artifact))

    assert len(follow_up_events) == 2
    assert follow_up_events[0] is artifact
    assert isinstance(follow_up_events[1], MessageChunkSSE)
    assert follow_up_events[1].data.delta == " After"
    assert not parser.has_pending_placeholder()
