"""Preprocessing of PDF content in tool result messages for LLM consumption."""

from __future__ import annotations

import logging
from typing import Any

from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    LlmClientFunctionCallResultMessage,
    RawObjectDataFormat,
    SingleDataContent,
    SingleFileReference,
)

logger = logging.getLogger(__name__)

# Type alias for data entry union
DataEntry = Any  # ClientCommandResult | DataContent | DataFileReferences | ...
PdfLikeItem = SingleDataContent | SingleFileReference


def _is_pdf_item(item: PdfLikeItem) -> bool:
    """Check if an item (content or reference) contains PDF data."""
    if item.data_format is None:
        return False
    return item.data_format.data_type == "pdf"


def _get_filename(item: PdfLikeItem) -> str:
    """Extract filename from an item's data_format."""
    if item.data_format is None:
        return "document.pdf"
    return getattr(item.data_format, "filename", None) or "document.pdf"


async def _extract_pdf_text(content: str, filename: str) -> str | None:
    """Extract text from PDF content (base64 or URL).

    Returns extracted text or None if extraction fails or dependencies missing.
    """
    try:
        from openbb_pydantic_ai.pdf import extract_pdf_content
    except ImportError:
        logger.debug("PDF dependencies not installed, skipping extraction")
        return None

    try:
        is_url = content.startswith(("http://", "https://"))
        result = await extract_pdf_content(
            url=content if is_url else None,
            content=content if not is_url else None,
            filename=filename,
        )
        return result.text
    except Exception as e:
        logger.warning("Failed to extract PDF content: %s", e)
        return None


async def preprocess_pdf_in_result(
    message: LlmClientFunctionCallResultMessage,
) -> LlmClientFunctionCallResultMessage:
    """Pre-process a result message, extracting text from any PDF items.

    Replaces PDF base64/URL content with extracted text so the LLM can read it.
    Items that fail extraction are left unchanged.

    Parameters
    ----------
    message : LlmClientFunctionCallResultMessage
        The result message potentially containing PDF data

    Returns
    -------
    LlmClientFunctionCallResultMessage
        A new message with PDF content replaced by extracted text,
        or the original message if no PDF items found
    """
    if not message.data:
        return message

    modified = False
    new_data: list[DataEntry] = []

    for data_content in message.data:
        # Handle direct content items
        if isinstance(data_content, DataContent) and data_content.items:
            new_items: list[SingleDataContent] = []
            for item in data_content.items:
                if not _is_pdf_item(item):
                    new_items.append(item)
                    continue

                filename = _get_filename(item)
                extracted = await _extract_pdf_text(item.content, filename)

                modified = True
                if extracted:
                    new_items.append(
                        SingleDataContent(
                            content=extracted,
                            data_format=RawObjectDataFormat(
                                data_type="object",
                                parse_as="text",
                            ),
                            citable=item.citable,
                        )
                    )
                else:
                    # Extraction failed — replace with a note instead of
                    # passing raw base64 PDF bytes to the LLM.
                    new_items.append(
                        SingleDataContent(
                            content=f"[PDF '{filename}' could not be extracted]",
                            data_format=RawObjectDataFormat(
                                data_type="object",
                                parse_as="text",
                            ),
                            citable=False,
                        )
                    )

            new_data.append(
                DataContent(
                    items=new_items,
                    extra_citations=data_content.extra_citations,
                )
            )
            continue

        # Handle file reference items
        if isinstance(data_content, DataFileReferences) and data_content.items:
            contains_pdf = any(_is_pdf_item(item) for item in data_content.items)
            if not contains_pdf:
                new_data.append(data_content)
                continue

            modified = True
            converted_items: list[SingleDataContent] = []
            for item in data_content.items:
                filename = _get_filename(item)

                if _is_pdf_item(item):
                    extracted = await _extract_pdf_text(str(item.url), filename)
                    if extracted:
                        converted_items.append(
                            SingleDataContent(
                                content=extracted,
                                data_format=RawObjectDataFormat(
                                    data_type="object",
                                    parse_as="text",
                                ),
                                citable=item.citable,
                            )
                        )
                        continue

                    # Extraction failed; fall back to URL as plain text to avoid
                    # passing raw PDF bytes to the LLM.
                    converted_items.append(
                        SingleDataContent(
                            content=str(item.url),
                            data_format=RawObjectDataFormat(
                                data_type="object",
                                parse_as="text",
                            ),
                            citable=item.citable,
                        )
                    )
                    continue

                # Non-PDF references: keep as textual reference for consistency
                converted_items.append(
                    SingleDataContent(
                        content=str(item.url),
                        data_format=RawObjectDataFormat(
                            data_type="object",
                            parse_as="text",
                        ),
                        citable=getattr(item, "citable", True),
                    )
                )

            new_data.append(
                DataContent(
                    items=converted_items,
                    extra_citations=data_content.extra_citations,
                )
            )
            continue

        # Pass through any other data types unchanged
        new_data.append(data_content)

    if not modified:
        return message

    # Create new message with processed data
    return LlmClientFunctionCallResultMessage(
        function=message.function,
        input_arguments=message.input_arguments,
        data=new_data,
        extra_state=message.extra_state,
    )


async def preprocess_pdf_in_results(
    results: list[LlmClientFunctionCallResultMessage],
) -> list[LlmClientFunctionCallResultMessage]:
    """Pre-process multiple result messages, extracting PDF text.

    Parameters
    ----------
    results : list[LlmClientFunctionCallResultMessage]
        List of result messages to process

    Returns
    -------
    list[LlmClientFunctionCallResultMessage]
        List of processed messages with PDF content extracted
    """
    processed: list[LlmClientFunctionCallResultMessage] = []
    for result in results:
        processed.append(await preprocess_pdf_in_result(result))
    return processed
