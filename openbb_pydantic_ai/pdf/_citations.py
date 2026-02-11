"""Citation bounding box extraction from docling provenance."""

from __future__ import annotations

from openbb_pydantic_ai.pdf._types import CitationBoundingBox, ProvenanceItem


def extract_citations_from_provenance(
    provenance_items: list[tuple[ProvenanceItem, str]],
    *,
    max_citations: int = 100,
) -> list[CitationBoundingBox]:
    """Convert ``(ProvenanceItem, text)`` pairs to citation bounding boxes.

    Each provenance item contains bbox coordinates (l, t, r, b) and a page
    number.  Docling uses TOPLEFT origin by default; coordinates are passed
    through as-is since the consumer handles coordinate system differences.

    Parameters
    ----------
    provenance_items
        List of ``(provenance_item, parent_text)`` tuples where the provenance
        item has ``bbox`` (with l/t/r/b) and ``page_no`` attributes, and
        ``parent_text`` is the source text from the owning ``DocItem``.
    max_citations
        Maximum number of citations to return, to avoid overwhelming the UI.
    """
    citations: list[CitationBoundingBox] = []

    for prov, text in provenance_items:
        if len(citations) >= max_citations:
            break

        bbox = getattr(prov, "bbox", None)
        page_no = getattr(prov, "page_no", None)

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
