"""Internal data types for PDF extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BoundingBox(Protocol):
    """Docling bounding box with left/top/right/bottom coordinates."""

    l: float  # noqa: E741
    t: float
    r: float
    b: float


class ProvenanceItem(Protocol):
    """Docling provenance item with bbox and page number."""

    bbox: BoundingBox | None
    page_no: int | None


class Caption(Protocol):
    """Docling caption with text content."""

    text: str | None


class DocItem(Protocol):
    """Docling document item with text and provenance."""

    text: str | None
    prov: list[ProvenanceItem] | None


class TableItem(DocItem, Protocol):
    """Docling table item with captions and markdown export."""

    captions: list[Caption] | None

    def export_to_markdown(self, *, doc: Any) -> str: ...


@dataclass(slots=True)
class PdfExtractionResult:
    """Result of PDF text extraction.

    Contains the extracted text and metadata.
    """

    text: str
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
