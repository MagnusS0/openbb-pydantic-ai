"""Citation bounding box extraction from docling provenance."""

from __future__ import annotations

from typing import Any

from openbb_pydantic_ai.pdf._types import CitationBoundingBox


def extract_citations_from_provenance(
    provenance_items: list[Any],
    *,
    max_citations: int = 100,
) -> list[CitationBoundingBox]:
    """
    Convert docling ProvenanceItem objects to citation bounding boxes.

    Each provenance item contains bbox coordinates (l, t, r, b) and a page number.
    Docling uses BOTTOMLEFT origin by default, but coordinates are passed through
    as-is since the consumer handles coordinate system differences.

    Parameters
    ----------
    provenance_items
        List of docling ProvenanceItem or similar objects with `bbox` and `page_no`.
    max_citations
        Maximum number of citations to return, to avoid overwhelming the UI.

    Returns
    -------
    list[CitationBoundingBox]
        Bounding boxes suitable for citation highlighting.
    """
    citations: list[CitationBoundingBox] = []

    for item in provenance_items:
        if len(citations) >= max_citations:
            break

        bbox = getattr(item, "bbox", None)
        page_no = getattr(item, "page_no", None)
        text = getattr(item, "text", "") or ""

        if bbox is None or page_no is None:
            continue

        citations.append(
            CitationBoundingBox(
                text=text,
                page=page_no,
                x0=getattr(bbox, "l", 0.0),
                top=getattr(bbox, "t", 0.0),
                x1=getattr(bbox, "r", 0.0),
                bottom=getattr(bbox, "b", 0.0),
            )
        )

    return citations
