"""Internal data types for PDF extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PdfExtractionResult:
    """Result of PDF text extraction.

    Contains the extracted text, provenance items for citations, and metadata.
    Provenance items are docling's internal type containing bounding box info.
    """

    text: str
    provenance: list[Any]
    metadata: dict[str, Any]


@dataclass(slots=True, frozen=True)
class CitationBoundingBox:
    """Bounding box for a citation highlight in a PDF.

    Coordinates are relative to the page. Page numbers are 1-indexed.
    """

    text: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float
