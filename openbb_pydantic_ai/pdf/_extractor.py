"""Core PDF extraction logic using docling."""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openbb_pydantic_ai.pdf._types import PdfExtractionResult

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument
    from docling.document_converter import DocumentConverter


def _collect_provenance(doc: DoclingDocument) -> list[Any]:
    """Collect all provenance items from a DoclingDocument."""
    provenance: list[Any] = []
    for item, _level in doc.iterate_items():
        prov_list = getattr(item, "prov", None)
        if prov_list:
            provenance.extend(prov_list)
    return provenance


@functools.lru_cache(maxsize=4)
def _get_converter(*, enable_ocr: bool, do_table_structure: bool) -> DocumentConverter:
    """Return a cached ``DocumentConverter`` to avoid reloading models per call."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions(
        do_ocr=enable_ocr,
        do_table_structure=do_table_structure,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _extract_sync(
    pdf_path: Path,
    *,
    enable_ocr: bool,
) -> tuple[str, list[Any], dict[str, Any], DoclingDocument]:
    """Synchronous PDF extraction using docling.

    Returns (text, provenance_items, metadata, document).
    """
    converter = _get_converter(enable_ocr=enable_ocr, do_table_structure=True)

    result = converter.convert(pdf_path)
    doc = result.document

    text = doc.export_to_markdown()
    provenance = _collect_provenance(doc)

    metadata: dict[str, Any] = {
        "page_count": doc.num_pages(),
        "filename": pdf_path.name,
    }

    return text, provenance, metadata, doc


async def _load_pdf_bytes(
    content: str | None,
    url: str | None,
) -> bytes:
    """Load bytes from base64 content or a remote URL."""
    if content is not None:
        return base64.b64decode(content)

    if url is None:
        msg = "Either 'content' or 'url' must be provided"
        raise ValueError(msg)

    import httpx

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


async def extract_pdf_content(
    content: str | None = None,
    url: str | None = None,
    *,
    filename: str = "document.pdf",
    enable_ocr: bool = False,
) -> PdfExtractionResult:
    """
    Extract text and provenance from a PDF.

    Handles both base64-encoded content and URL downloads. Uses docling's
    DocumentConverter with table extraction enabled. OCR is disabled by
    default for speed — most text-based PDFs extract accurately without it.

    Parameters
    ----------
    content
        Base64-encoded PDF content (mutually exclusive with `url`).
    url
        URL to download the PDF from (mutually exclusive with `content`).
    filename
        Filename for metadata and temporary file naming.
    enable_ocr
        Whether to enable OCR for scanned documents. Disabled by default
        because it is significantly slower. Enable for scanned-image PDFs.

    Returns
    -------
    PdfExtractionResult
        Extracted text, provenance items, and metadata.

    Raises
    ------
    ValueError
        If both or neither of `content` and `url` are provided.
    """
    if (content is None) == (url is None):
        msg = "Exactly one of 'content' or 'url' must be provided"
        raise ValueError(msg)

    pdf_bytes = await _load_pdf_bytes(content, url)
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(pdf_bytes)

        loop = asyncio.get_running_loop()
        text, provenance, metadata, _doc = await loop.run_in_executor(
            None,
            functools.partial(_extract_sync, pdf_path, enable_ocr=enable_ocr),
        )

        if url:
            metadata["url"] = url
        metadata["doc_id"] = doc_id

    return PdfExtractionResult(
        text=text,
        provenance=provenance,
        metadata=metadata,
    )


async def extract_pdf_document(
    content: str | None = None,
    url: str | None = None,
    *,
    filename: str = "document.pdf",
    enable_ocr: bool = False,
) -> tuple[PdfExtractionResult, DoclingDocument]:
    """Extract PDF and return both extraction result and ``DoclingDocument``."""
    if (content is None) == (url is None):
        msg = "Exactly one of 'content' or 'url' must be provided"
        raise ValueError(msg)

    pdf_bytes = await _load_pdf_bytes(content, url)
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / filename
        pdf_path.write_bytes(pdf_bytes)

        loop = asyncio.get_running_loop()
        text, provenance, metadata, doc = await loop.run_in_executor(
            None,
            functools.partial(_extract_sync, pdf_path, enable_ocr=enable_ocr),
        )

        if url:
            metadata["url"] = url
        metadata["doc_id"] = doc_id

    return (
        PdfExtractionResult(
            text=text,
            provenance=provenance,
            metadata=metadata,
        ),
        doc,
    )
