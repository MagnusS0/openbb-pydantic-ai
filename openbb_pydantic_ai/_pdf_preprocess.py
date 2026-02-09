"""PDF preprocessing wrapper that handles optional dependency gracefully."""

from __future__ import annotations

from openbb_ai.models import LlmClientFunctionCallResultMessage, LlmMessage


async def preprocess_pdf_in_results(
    results: list[LlmClientFunctionCallResultMessage],
) -> list[LlmClientFunctionCallResultMessage]:
    """Pre-process result messages, extracting PDF text where possible.

    This wrapper handles the case where PDF dependencies are not installed
    by returning the original results unchanged.

    Parameters
    ----------
    results : list[LlmClientFunctionCallResultMessage]
        List of result messages to process

    Returns
    -------
    list[LlmClientFunctionCallResultMessage]
        List of processed messages with PDF content extracted,
        or original messages if PDF deps not installed
    """
    try:
        from openbb_pydantic_ai.pdf._preprocess import (
            preprocess_pdf_in_results as _preprocess,
        )
    except ImportError:
        # PDF dependencies not installed, return unchanged
        return results

    return await _preprocess(results)


async def preprocess_pdf_in_messages(messages: list[LlmMessage]) -> list[LlmMessage]:
    """Pre-process PDF data inside mixed history messages.

    Only ``LlmClientFunctionCallResultMessage`` entries are inspected and
    transformed. Other message types are returned unchanged.
    """
    result_indices: list[int] = []
    results: list[LlmClientFunctionCallResultMessage] = []

    for idx, message in enumerate(messages):
        if isinstance(message, LlmClientFunctionCallResultMessage):
            result_indices.append(idx)
            results.append(message)

    if not results:
        return messages

    processed_results = await preprocess_pdf_in_results(results)
    if len(processed_results) != len(results):
        return messages

    changed = False
    merged = list(messages)
    for idx, processed in zip(result_indices, processed_results, strict=False):
        if merged[idx] is not processed:
            changed = True
        merged[idx] = processed

    return merged if changed else messages
