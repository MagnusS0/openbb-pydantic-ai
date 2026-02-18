"""Content serialization utilities for OpenBB Pydantic AI adapter."""

from __future__ import annotations

from typing import Any, cast

from openbb_ai.models import LlmClientFunctionCallResultMessage
from pydantic_core import from_json as _from_json
from pydantic_core import to_json as _pydantic_to_json
from pydantic_core import to_jsonable_python

from openbb_pydantic_ai._types import SerializedContent


def serialize_result(
    message: LlmClientFunctionCallResultMessage,
) -> SerializedContent:
    """Serialize a function call result message into a content dictionary.

    Parameters
    ----------
    message : LlmClientFunctionCallResultMessage
        The function call result message to serialize

    Returns
    -------
    SerializedContent
        A typed dictionary containing input_arguments, data, and
        optionally extra_state
    """
    data: list[Any] = [
        to_jsonable_python(item, exclude_none=True) for item in message.data
    ]

    content: SerializedContent = cast(
        SerializedContent,
        {
            "input_arguments": message.input_arguments,
            "data": data,
        },
    )
    if message.extra_state:
        content["extra_state"] = message.extra_state
    return content


def parse_json(raw_content: str) -> Any:
    """Parse JSON content, returning the original string if parsing fails.

    Parameters
    ----------
    raw_content : str
        The raw JSON string to parse

    Returns
    -------
    Any
        Parsed JSON object or original string if parsing fails
    """
    try:
        return _from_json(raw_content, cache_strings="keys")
    except ValueError:
        return raw_content


def to_string(content: Any) -> str | None:
    """Convert content to string with JSON fallback.

    Parameters
    ----------
    content : Any
        Content to stringify

    Returns
    -------
    str | None
        String representation or None if content is None
    """
    if content is None:
        return None
    if isinstance(content, str):
        return content
    try:
        return _pydantic_to_json(content, serialize_unknown=True).decode()
    except Exception:
        return str(content)


def to_json(value: Any) -> str:
    """Convert value to JSON string with fallback to str().

    Parameters
    ----------
    value : Any
        Value to convert to JSON

    Returns
    -------
    str
        JSON string representation
    """
    try:
        return _pydantic_to_json(value, serialize_unknown=True).decode()
    except Exception:
        return str(value)
