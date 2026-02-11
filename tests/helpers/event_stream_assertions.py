from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from openbb_ai.models import StatusUpdateSSE


def collect_events(async_iterable: Any) -> list[Any]:
    async def _gather() -> list[Any]:
        return [event async for event in async_iterable]

    return asyncio.run(_gather())


def find_status_with_artifacts(events: Iterable[Any]) -> StatusUpdateSSE:
    for event in events:
        if isinstance(event, StatusUpdateSSE) and event.data.artifacts:
            return event
    raise AssertionError("Expected status update with artifacts")
