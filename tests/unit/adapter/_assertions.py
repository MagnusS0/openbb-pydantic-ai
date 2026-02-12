from __future__ import annotations

from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from openbb_pydantic_ai import OpenBBAIAdapter


def tool_call_parts(adapter: OpenBBAIAdapter) -> list[ToolCallPart]:
    return [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolCallPart)
    ]


def tool_return_parts(adapter: OpenBBAIAdapter) -> list[ToolReturnPart]:
    return [
        part
        for message in adapter.messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]


def visible_turn_text(adapter: OpenBBAIAdapter) -> list[str]:
    turns: list[str] = []
    for msg in adapter.messages:
        parts_text = [
            part.content
            for part in msg.parts
            if isinstance(part, (UserPromptPart, TextPart))
        ]
        if parts_text:
            turns.append(" ".join(str(text) for text in parts_text))
    return turns
