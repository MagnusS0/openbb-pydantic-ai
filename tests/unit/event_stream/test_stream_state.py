from __future__ import annotations

from openbb_ai.models import Citation, SourceInfo

from openbb_pydantic_ai._event_stream_components import StreamState


def _citation(doc_id: str) -> Citation:
    return Citation(
        source_info=SourceInfo(
            type="direct retrieval",
            name=f"{doc_id}.pdf",
            metadata={"doc_id": doc_id},
            citable=True,
        ),
        details=[{"doc_id": doc_id}],
    )


def test_drain_citations_returns_and_clears_state() -> None:
    state = StreamState()
    state.add_citation(_citation("a"))
    state.add_citation(_citation("b"))

    drained = state.drain_citations()

    assert [item.source_info.metadata.get("doc_id") for item in drained] == ["a", "b"]
    assert not state.has_citations()
    assert state.drain_citations() == []


def test_drain_unflushed_local_entries_clears_after_each_drain() -> None:
    state = StreamState()

    state.register_local_tool_call(tool_call_id="call-1", tool_name="tool", args={})
    state.complete_local_tool_call("call-1", {"result": 1})
    first = state.drain_unflushed_local_entries()
    assert [entry.tool_call_id for entry in first] == ["call-1"]
    assert state.local_completed_entries == []

    state.register_local_tool_call(tool_call_id="call-2", tool_name="tool", args={})
    state.complete_local_tool_call("call-2", {"result": 2})
    second = state.drain_unflushed_local_entries()

    assert [entry.tool_call_id for entry in second] == ["call-2"]
    assert state.drain_unflushed_local_entries() == []
    assert state.local_completed_entries == []
