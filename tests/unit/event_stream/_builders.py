from __future__ import annotations

from typing import Any

from openbb_ai.models import (
    DataContent,
    LlmClientFunctionCallResultMessage,
    SingleDataContent,
)


def result_message(
    function: str,
    input_args: dict[str, Any],
) -> LlmClientFunctionCallResultMessage:
    return LlmClientFunctionCallResultMessage(
        function=function,
        input_arguments=input_args,
        data=[
            DataContent(
                items=[SingleDataContent(content='[{"value": 1}]')],
            )
        ],
    )


def raw_object_item(
    content: str,
    *,
    parse_as: str = "table",
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "content": content,
        "data_format": {
            "data_type": "object",
            "parse_as": parse_as,
        },
    }
    if name:
        item["name"] = name
    if description:
        item["description"] = description
    return item
