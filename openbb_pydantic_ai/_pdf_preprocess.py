"""PDF preprocessing wrapper that handles optional dependency gracefully."""

from __future__ import annotations

from openbb_ai.models import LlmClientFunctionCallResultMessage


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
