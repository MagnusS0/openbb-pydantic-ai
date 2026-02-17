"""Tests for PDF preprocessing in tool result messages."""

from __future__ import annotations

import json

from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    LlmClientFunctionCallResultMessage,
    PdfDataFormat,
    SingleDataContent,
    SingleFileReference,
)
from pytest_mock import MockerFixture

from openbb_pydantic_ai.pdf import _preprocess as pre
from openbb_pydantic_ai.pdf._preprocess import (
    preprocess_pdf_in_result,
    preprocess_pdf_in_results,
)


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


async def test_preprocess_injects_pdf_toc(mocker: MockerFixture) -> None:
    """PDF content is replaced with a TOC prompt for progressive retrieval."""

    toc = (
        "Document has been extracted, here is the table of contents:\n\n"
        "doc_id: abc123\n"
        "Use the `pdf_query` tool."
    )
    mocker.patch(
        "openbb_pydantic_ai.pdf._preprocess._extract_pdf_toc",
        return_value=toc,
    )
    message = _make_pdf_result("JVBERi0xLjcK")
    processed = await preprocess_pdf_in_result(message)

    # Should have injected TOC prompt
    assert processed is not message  # New message created
    assert processed.data
    data_content = processed.data[0]
    assert isinstance(data_content, DataContent)
    assert data_content.items
    item = data_content.items[0]
    # Content should be TOC text, not raw base64
    assert not item.content.startswith("JVBERi")  # Not base64 PDF header
    assert "table of contents" in item.content.lower()
    assert "doc_id:" in item.content
    assert "pdf_query" in item.content


async def test_preprocess_preserves_non_pdf_content() -> None:
    """Non-PDF content is left unchanged."""

    message = _make_text_result('{"data": "test"}')
    processed = await preprocess_pdf_in_result(message)

    # Should return same message (no modification)
    assert processed is message


async def test_preprocess_handles_empty_data() -> None:
    """Empty data list is handled gracefully."""

    message = LlmClientFunctionCallResultMessage(
        function="test",
        input_arguments={},
        data=[],
    )
    processed = await preprocess_pdf_in_result(message)

    assert processed is message


async def test_preprocess_multiple_results(mocker: MockerFixture) -> None:
    """Multiple result messages are all processed."""

    mocker.patch(
        "openbb_pydantic_ai.pdf._preprocess._extract_pdf_toc",
        return_value="TOC",
    )
    results = [
        _make_pdf_result("JVBERi0xLjcK"),
        _make_text_result("plain text"),
    ]
    processed = await preprocess_pdf_in_results(results)

    assert len(processed) == 2
    # First should be modified (PDF extracted)
    assert processed[0] is not results[0]
    # Second should be unchanged (not PDF)
    assert processed[1] is results[1]


async def test_preprocess_converts_pdf_file_reference_to_toc(
    mocker: MockerFixture, pdf_data_format: dict[str, str]
) -> None:
    """PDF file references are fetched and converted to TOC text content."""

    message = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={"widget_uuid": "test-uuid"},
        data=[
            DataFileReferences(
                items=[
                    SingleFileReference(
                        url="https://example.com/test.pdf",  # type: ignore[arg-type]
                        data_format=PdfDataFormat(**pdf_data_format),  # type: ignore[arg-type]
                    )
                ]
            )
        ],
    )

    mocker.patch.object(pre, "_extract_pdf_toc", return_value="toc from url")

    processed = await pre.preprocess_pdf_in_result(message)

    assert processed is not message
    assert processed.data
    data_content = processed.data[0]
    assert isinstance(data_content, DataContent)
    assert data_content.items
    item = data_content.items[0]
    assert isinstance(item, SingleDataContent)
    assert item.content == "toc from url"
    assert item.data_format.parse_as == "text"


async def test_preprocess_converts_embedded_pdf_json_and_registers_widget_aliases(
    mocker: MockerFixture,
) -> None:
    """PDF metadata embedded in JSON content should still become TOC."""

    widget_id = "file-c0ebd61e-39fa-4382-b82f-4374ea96a30f"
    widget_uuid = "f5638ae4-5b83-49f8-8b02-925cd947646a"
    pdf_url = (
        "https://example.com/nvda.pdf"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=token"
    )
    message = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={
            "data_sources": [
                {
                    "widget_uuid": widget_uuid,
                    "id": widget_id,
                    "origin": "OpenBB Hub",
                    "input_args": {},
                }
            ]
        },
        data=[
            DataContent(
                items=[
                    SingleDataContent(
                        content=json.dumps(
                            {
                                "url": pdf_url,
                                "data_format": {
                                    "data_type": "pdf",
                                    "filename": "NVDA%20filing.pdf",
                                },
                                "content": None,
                            }
                        ),
                    )
                ]
            )
        ],
    )

    extract_mock = mocker.patch.object(
        pre, "_extract_pdf_toc", return_value="toc from json"
    )
    mocker.patch.object(pre, "_doc_id_for_source", return_value="canonical-doc-id")
    register_mock = mocker.patch.object(pre, "register_document_source")

    processed = await pre.preprocess_pdf_in_result(message)

    assert processed is not message
    assert processed.data
    data_content = processed.data[0]
    assert isinstance(data_content, DataContent)
    assert data_content.items
    item = data_content.items[0]
    assert item.content == "toc from json"
    assert item.data_format.parse_as == "text"

    extract_mock.assert_called_once_with(pdf_url, "NVDA filing.pdf")
    aliases = {call.args[1] for call in register_mock.call_args_list}
    assert widget_id in aliases
    assert "c0ebd61e-39fa-4382-b82f-4374ea96a30f" in aliases
    assert widget_uuid in aliases


async def test_get_extraction_lock_returns_same_lock_for_same_key() -> None:
    pre._extraction_locks.clear()
    key = "doc-key"

    lock_a = await pre._get_extraction_lock(key)
    lock_b = await pre._get_extraction_lock(key)

    assert lock_a is lock_b
    assert key in pre._extraction_locks
