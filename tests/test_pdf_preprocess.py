"""Tests for PDF preprocessing in tool result messages."""

from __future__ import annotations

import pytest
from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    LlmClientFunctionCallResultMessage,
    PdfDataFormat,
    SingleDataContent,
    SingleFileReference,
)
from pytest_mock import MockerFixture


def _make_pdf_result(
    pdf_content: str, filename: str = "test.pdf"
) -> LlmClientFunctionCallResultMessage:
    """Create a result message with PDF content."""
    return LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"widget_uuid": "test-uuid"},
        data=[
            DataContent(
                items=[
                    SingleDataContent(
                        content=pdf_content,
                        data_format=PdfDataFormat(
                            data_type="pdf",
                            filename=filename,
                        ),
                    )
                ]
            )
        ],
    )


def _make_text_result(text: str) -> LlmClientFunctionCallResultMessage:
    """Create a result message with text content."""
    return LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"widget_uuid": "test-uuid"},
        data=[
            DataContent(
                items=[
                    SingleDataContent(
                        content=text,
                    )
                ]
            )
        ],
    )


@pytest.mark.anyio
async def test_preprocess_extracts_pdf_text(sample_pdf_base64: str) -> None:
    """PDF content is extracted and replaced with text."""
    from openbb_pydantic_ai.pdf._preprocess import preprocess_pdf_in_result

    message = _make_pdf_result(sample_pdf_base64)
    processed = await preprocess_pdf_in_result(message)

    # Should have extracted text
    assert processed is not message  # New message created
    assert processed.data
    data_content = processed.data[0]
    assert isinstance(data_content, DataContent)
    assert data_content.items
    item = data_content.items[0]
    # Content should be extracted text, not base64
    assert not item.content.startswith("JVBERi")  # Not base64 PDF header
    assert "PDF" in item.content or "Test" in item.content  # Has readable text


@pytest.mark.anyio
async def test_preprocess_preserves_non_pdf_content() -> None:
    """Non-PDF content is left unchanged."""
    from openbb_pydantic_ai.pdf._preprocess import preprocess_pdf_in_result

    message = _make_text_result('{"data": "test"}')
    processed = await preprocess_pdf_in_result(message)

    # Should return same message (no modification)
    assert processed is message


@pytest.mark.anyio
async def test_preprocess_handles_empty_data() -> None:
    """Empty data list is handled gracefully."""
    from openbb_pydantic_ai.pdf._preprocess import preprocess_pdf_in_result

    message = LlmClientFunctionCallResultMessage(
        function="test",
        input_arguments={},
        data=[],
    )
    processed = await preprocess_pdf_in_result(message)

    assert processed is message


@pytest.mark.anyio
async def test_preprocess_multiple_results(sample_pdf_base64: str) -> None:
    """Multiple result messages are all processed."""
    from openbb_pydantic_ai.pdf._preprocess import preprocess_pdf_in_results

    results = [
        _make_pdf_result(sample_pdf_base64),
        _make_text_result("plain text"),
    ]
    processed = await preprocess_pdf_in_results(results)

    assert len(processed) == 2
    # First should be modified (PDF extracted)
    assert processed[0] is not results[0]
    # Second should be unchanged (not PDF)
    assert processed[1] is results[1]


@pytest.mark.anyio
async def test_preprocess_extracts_pdf_file_reference(
    mocker: MockerFixture, pdf_data_format: dict[str, str]
) -> None:
    """PDF file references are fetched and converted to text content."""
    from openbb_pydantic_ai.pdf import _preprocess as pre

    message = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"widget_uuid": "test-uuid"},
        data=[
            DataFileReferences(
                items=[
                    SingleFileReference(
                        url="https://example.com/test.pdf",
                        data_format=PdfDataFormat(**pdf_data_format),
                    )
                ]
            )
        ],
    )

    mocker.patch.object(pre, "_extract_pdf_text", return_value="extracted from url")

    processed = await pre.preprocess_pdf_in_result(message)

    assert processed is not message
    assert processed.data
    data_content = processed.data[0]
    assert isinstance(data_content, DataContent)
    assert data_content.items
    item = data_content.items[0]
    assert isinstance(item, SingleDataContent)
    assert item.content == "extracted from url"
    assert item.data_format.parse_as == "text"
