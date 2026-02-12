"""
Optional PDF extraction support for openbb-pydantic-ai.

Install with: pip install "openbb-pydantic-ai[pdf]"

This module provides PDF text extraction using docling, with support for
OCR and citation bounding boxes via provenance tracking.
"""

from __future__ import annotations

try:
    from docling.datamodel.base_models import InputFormat as _InputFormat
    from docling.document_converter import DocumentConverter as _DocumentConverter

    del _InputFormat, _DocumentConverter
except ImportError as _import_error:
    raise ImportError(
        "Please install docling to use PDF processing. "
        'Install with: pip install "openbb-pydantic-ai[pdf]"'
    ) from _import_error

from openbb_pydantic_ai.pdf._citations import extract_citations_from_provenance
from openbb_pydantic_ai.pdf._extractor import extract_pdf_content, extract_pdf_document
from openbb_pydantic_ai.pdf._types import CitationBoundingBox, PdfExtractionResult

__all__ = [
    "CitationBoundingBox",
    "PdfExtractionResult",
    "extract_citations_from_provenance",
    "extract_pdf_content",
    "extract_pdf_document",
]
