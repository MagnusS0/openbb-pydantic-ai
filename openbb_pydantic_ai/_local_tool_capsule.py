"""Codec helpers for stateless local tool history capsules."""

from __future__ import annotations

import base64
import zlib
from typing import Any, Sequence

from pydantic import BaseModel, Field


class LocalToolEntry(BaseModel):
    """Single local tool call/return entry captured in a capsule."""

    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None


class LocalToolState(BaseModel):
    """Packed local tool state persisted in ``extra_state``."""

    entries: list[LocalToolEntry] = Field(default_factory=list)

    def pack(self) -> str:
        raw = self.model_dump_json().encode()
        return base64.b85encode(zlib.compress(raw, 9)).decode("ascii")

    @classmethod
    def unpack(cls, packed: str) -> LocalToolState:
        if not isinstance(packed, str) or not packed:
            raise ValueError("Capsule payload must be a non-empty string.")
        return cls.model_validate_json(zlib.decompress(base64.b85decode(packed)))


def pack_tool_history(entries: Sequence[LocalToolEntry]) -> str:
    """Serialize tool entries into a compact ASCII payload."""
    return LocalToolState(entries=list(entries)).pack()


def unpack_tool_history(packed: str) -> list[LocalToolEntry]:
    """Decode tool history payload from ``extra_state``."""
    return LocalToolState.unpack(packed).entries
