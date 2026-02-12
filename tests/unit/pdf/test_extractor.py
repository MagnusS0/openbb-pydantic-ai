"""Tests for PDF extraction functionality."""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from openbb_pydantic_ai.pdf import _extractor as extractor
from openbb_pydantic_ai.pdf._extractor import extract_pdf_content


class _LoopStub:
    def __init__(self, payload):
        self.payload = payload

    async def run_in_executor(self, _executor, _func):
        return self.payload


async def test_extract_pdf_requires_exactly_one_input() -> None:
    """Raises ValueError when both or neither content/url provided."""

    with pytest.raises(ValueError, match="Exactly one"):
        await extract_pdf_content(content="abc", url="http://example.com/test.pdf")

    with pytest.raises(ValueError, match="Exactly one"):
        await extract_pdf_content()


async def test_extract_pdf_returns_metadata(mocker) -> None:
    """Metadata includes page count, filename, and stable doc_id."""

    pdf_bytes = b"fake-pdf-bytes"
    loop = _LoopStub(
        ("markdown", {"page_count": 1, "filename": "document.pdf"}, object())
    )
    mocker.patch.object(
        extractor,
        "_load_pdf_bytes",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    )
    mocker.patch.object(extractor.asyncio, "get_running_loop", return_value=loop)

    result = await extractor.extract_pdf_content(
        content="ZmFrZS1iYXNlNjQ=",
        filename="document.pdf",
        enable_ocr=False,
    )

    assert "page_count" in result.metadata
    assert result.metadata["filename"] == "document.pdf"
    assert "doc_id" in result.metadata
    assert result.metadata["doc_id"] == hashlib.sha256(pdf_bytes).hexdigest()


async def test_extract_pdf_returns_text(mocker) -> None:
    """Text is extracted from PDF content."""

    loop = _LoopStub(("markdown", {"page_count": 1, "filename": "test.pdf"}, object()))
    mocker.patch.object(
        extractor,
        "_load_pdf_bytes",
        new_callable=AsyncMock,
        return_value=b"pdf",
    )
    mocker.patch.object(extractor.asyncio, "get_running_loop", return_value=loop)

    result = await extractor.extract_pdf_content(
        content="cGRm",
        filename="test.pdf",
        enable_ocr=False,
    )

    assert result.text == "markdown"


async def test_extract_pdf_document_returns_doc(mocker) -> None:
    """`extract_pdf_document` should return both result and docling document."""

    doc = object()
    pdf_bytes = b"doc-pdf-bytes"
    loop = _LoopStub(("markdown", {"page_count": 2, "filename": "test.pdf"}, doc))
    mocker.patch.object(
        extractor,
        "_load_pdf_bytes",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    )
    mocker.patch.object(extractor.asyncio, "get_running_loop", return_value=loop)

    result, returned_doc = await extractor.extract_pdf_document(
        content="ZG9j",
        filename="test.pdf",
        enable_ocr=False,
    )

    assert result.metadata["filename"] == "test.pdf"
    assert result.metadata["doc_id"]
    assert result.metadata["doc_id"] == hashlib.sha256(pdf_bytes).hexdigest()
    assert returned_doc is doc
