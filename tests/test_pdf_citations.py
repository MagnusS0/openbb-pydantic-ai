"""Tests for PDF citation bounding box extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openbb_pydantic_ai.pdf._citations import extract_citations_from_provenance
from openbb_pydantic_ai.pdf._types import CitationBoundingBox


@dataclass
class MockBBox:
    """Mock bounding box matching docling's structure."""

    l: float  # noqa: E741
    t: float
    r: float
    b: float


@dataclass
class MockProvenanceItem:
    """Mock provenance item matching docling's structure."""

    page_no: int
    bbox: MockBBox
    text: str = ""


def test_extract_citations_from_provenance_basic() -> None:
    """Basic provenance to citation conversion."""
    items: list[Any] = [
        MockProvenanceItem(
            page_no=1,
            bbox=MockBBox(l=10.0, t=20.0, r=100.0, b=40.0),
            text="Hello",
        ),
        MockProvenanceItem(
            page_no=2,
            bbox=MockBBox(l=50.0, t=100.0, r=200.0, b=120.0),
            text="World",
        ),
    ]

    citations = extract_citations_from_provenance(items)

    assert len(citations) == 2
    assert citations[0] == CitationBoundingBox(
        text="Hello",
        page=1,
        x0=10.0,
        top=20.0,
        x1=100.0,
        bottom=40.0,
    )
    assert citations[1] == CitationBoundingBox(
        text="World",
        page=2,
        x0=50.0,
        top=100.0,
        x1=200.0,
        bottom=120.0,
    )


def test_extract_citations_max_limit() -> None:
    """Respects max_citations limit."""
    items: list[Any] = [
        MockProvenanceItem(
            page_no=i,
            bbox=MockBBox(l=0.0, t=0.0, r=100.0, b=100.0),
            text=f"Item {i}",
        )
        for i in range(10)
    ]

    citations = extract_citations_from_provenance(items, max_citations=3)

    assert len(citations) == 3
    assert citations[0].text == "Item 0"
    assert citations[2].text == "Item 2"


def test_extract_citations_missing_bbox() -> None:
    """Skips items without bbox."""

    @dataclass
    class NoBBox:
        page_no: int
        text: str = ""

    items: list[Any] = [
        NoBBox(page_no=1, text="No bbox"),
        MockProvenanceItem(
            page_no=2,
            bbox=MockBBox(l=10.0, t=20.0, r=100.0, b=40.0),
            text="Has bbox",
        ),
    ]

    citations = extract_citations_from_provenance(items)

    assert len(citations) == 1
    assert citations[0].text == "Has bbox"


def test_extract_citations_missing_page_no() -> None:
    """Skips items without page_no."""

    @dataclass
    class NoPage:
        bbox: MockBBox
        text: str = ""

    items: list[Any] = [
        NoPage(bbox=MockBBox(l=10.0, t=20.0, r=100.0, b=40.0), text="No page"),
        MockProvenanceItem(
            page_no=1,
            bbox=MockBBox(l=50.0, t=60.0, r=150.0, b=80.0),
            text="Has page",
        ),
    ]

    citations = extract_citations_from_provenance(items)

    assert len(citations) == 1
    assert citations[0].text == "Has page"


def test_extract_citations_empty_text() -> None:
    """Handles items with empty or missing text."""
    items: list[Any] = [
        MockProvenanceItem(
            page_no=1,
            bbox=MockBBox(l=10.0, t=20.0, r=100.0, b=40.0),
            text="",
        ),
    ]

    citations = extract_citations_from_provenance(items)

    assert len(citations) == 1
    assert citations[0].text == ""


def test_extract_citations_empty_list() -> None:
    """Returns empty list for empty input."""
    citations = extract_citations_from_provenance([])
    assert citations == []
