"""Tests for PDF extraction functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openbb_pydantic_ai.pdf._types import PdfExtractionResult

docling = pytest.importorskip("docling", reason="docling not installed")


@pytest.mark.anyio
async def test_extract_pdf_from_base64(sample_pdf_base64: str) -> None:
    """Extract text from base64-encoded PDF."""
    from openbb_pydantic_ai.pdf import extract_pdf_content

    result: PdfExtractionResult = await extract_pdf_content(
        content=sample_pdf_base64,
        filename="test.pdf",
        enable_ocr=False,
    )

    assert result.text
    assert result.metadata["filename"] == "test.pdf"


@pytest.mark.anyio
async def test_extract_pdf_requires_exactly_one_input() -> None:
    """Raises ValueError when both or neither content/url provided."""
    from openbb_pydantic_ai.pdf import extract_pdf_content

    with pytest.raises(ValueError, match="Exactly one"):
        await extract_pdf_content(content="abc", url="http://example.com/test.pdf")

    with pytest.raises(ValueError, match="Exactly one"):
        await extract_pdf_content()


@pytest.mark.anyio
async def test_extract_pdf_returns_metadata(sample_pdf_base64: str) -> None:
    """Metadata includes page count and filename."""
    from openbb_pydantic_ai.pdf import extract_pdf_content

    result = await extract_pdf_content(
        content=sample_pdf_base64,
        filename="document.pdf",
        enable_ocr=False,
    )

    assert "page_count" in result.metadata
    assert result.metadata["filename"] == "document.pdf"


@pytest.mark.anyio
async def test_extract_pdf_returns_provenance(sample_pdf_base64: str) -> None:
    """Provenance list is returned for citation support."""
    from openbb_pydantic_ai.pdf import extract_pdf_content

    result = await extract_pdf_content(
        content=sample_pdf_base64,
        filename="test.pdf",
        enable_ocr=False,
    )

    assert isinstance(result.provenance, list)
