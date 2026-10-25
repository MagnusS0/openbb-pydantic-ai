"""Codec helpers for stateless local tool history capsules."""

from __future__ import annotations

import base64
import logging
import zlib
from typing import Any, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_PACKED_SIZE = 250_000
MAX_UNPACKED_SIZE = 2_000_000


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

    @staticmethod
    def _decompress_with_limit(compressed: bytes) -> bytes:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, MAX_UNPACKED_SIZE + 1)
        if len(raw) > MAX_UNPACKED_SIZE:
            raise ValueError("Capsule payload exceeds maximum unpacked size.")
        if not decompressor.eof:
            raise ValueError("Capsule payload is incomplete or malformed.")
        return raw

    @classmethod
    def unpack(cls, packed: str) -> LocalToolState:
        if not isinstance(packed, str) or not packed:
            raise ValueError("Capsule payload must be a non-empty string.")
        if len(packed) > MAX_PACKED_SIZE:
            raise ValueError("Capsule payload exceeds maximum packed size.")

        compressed = base64.b85decode(packed)
        raw = cls._decompress_with_limit(compressed)
        return cls.model_validate_json(raw)


def pack_tool_history(entries: Sequence[LocalToolEntry]) -> str:
    """Serialize tool entries into a compact ASCII payload.

    Oldest entries are dropped until the payload fits both ``MAX_PACKED_SIZE``
    and ``MAX_UNPACKED_SIZE`` so the result can always be unpacked again.
    """
    kept = list(entries)
    dropped = 0
    while True:
        state = LocalToolState(entries=kept)
        if len(state.model_dump_json()) <= MAX_UNPACKED_SIZE:
            packed = state.pack()
            if len(packed) <= MAX_PACKED_SIZE or not kept:
                break
        elif not kept:
            packed = state.pack()
            break
        kept.pop(0)
        dropped += 1
    if dropped:
        logger.warning(
            "Local tool capsule exceeded its size limits; dropped %d oldest entries.",
            dropped,
        )
    return packed


def unpack_tool_history(packed: str) -> list[LocalToolEntry]:
    """Decode tool history payload from ``extra_state``."""
    return LocalToolState.unpack(packed).entries
