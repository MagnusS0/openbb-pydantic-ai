from __future__ import annotations

import logging
import os

import pytest

from openbb_pydantic_ai import _local_tool_capsule as capsule
from openbb_pydantic_ai._local_tool_capsule import (
    LocalToolEntry,
    pack_tool_history,
    unpack_tool_history,
)

pytestmark = pytest.mark.regression_contract


def _entry(idx: int, size: int) -> LocalToolEntry:
    # Random hex does not compress, so packed size tracks ``size``.
    return LocalToolEntry(
        tool_call_id=f"call-{idx}",
        tool_name="tool",
        result=os.urandom(size // 2).hex(),
    )


def test_pack_drops_oldest_entries_to_fit_unpack_limit(caplog) -> None:
    entries = [_entry(i, 120_000) for i in range(4)]

    with caplog.at_level(logging.WARNING, logger=capsule.__name__):
        packed = pack_tool_history(entries)

    assert len(packed) <= capsule.MAX_PACKED_SIZE
    restored = unpack_tool_history(packed)
    assert restored
    assert [e.tool_call_id for e in restored] == [
        e.tool_call_id for e in entries[-len(restored) :]
    ]
    assert len(restored) < len(entries)
    assert "dropped" in caplog.text


def test_pack_drops_entries_exceeding_unpacked_limit(caplog) -> None:
    # Highly compressible results fit MAX_PACKED_SIZE but not MAX_UNPACKED_SIZE.
    entries = [
        LocalToolEntry(tool_call_id=f"call-{i}", tool_name="tool", result="a" * 900_000)
        for i in range(3)
    ]

    with caplog.at_level(logging.WARNING, logger=capsule.__name__):
        packed = pack_tool_history(entries)

    restored = unpack_tool_history(packed)
    assert [e.tool_call_id for e in restored] == ["call-1", "call-2"]
    assert "dropped 1 oldest" in caplog.text


def test_pack_small_payload_is_untouched(caplog) -> None:
    entries = [_entry(0, 100)]

    with caplog.at_level(logging.WARNING, logger=capsule.__name__):
        packed = pack_tool_history(entries)

    assert unpack_tool_history(packed) == entries
    assert not caplog.records
