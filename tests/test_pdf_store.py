"""Tests for in-memory PDF document store and TOC generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _BBox:
    l: float  # noqa: E741 - matches Docling bbox attribute names
    t: float
    r: float
    b: float


@dataclass
class _ProvenanceItem:
    page_no: int
    bbox: _BBox
    text: str = ""


class SectionHeaderItem:
    def __init__(self, text: str, level: int, page_no: int) -> None:
        self.text = text
        self.level = level
        self.prov = [
            _ProvenanceItem(page_no=page_no, bbox=_BBox(0, 0, 1, 1), text=text)
        ]


class TextItem:
    def __init__(self, text: str, page_no: int) -> None:
        self.text = text
        self.prov = [
            _ProvenanceItem(page_no=page_no, bbox=_BBox(0, 0, 1, 1), text=text)
        ]


class _FakeTable:
    def __init__(self, markdown: str, page_no: int) -> None:
        self._markdown = markdown
        self.prov = [
            _ProvenanceItem(
                page_no=page_no,
                bbox=_BBox(0, 0, 1, 1),
                text="table",
            )
        ]
        self.captions: list[Any] = []

    def export_to_markdown(self, doc=None) -> str:
        del doc
        return self._markdown


class _FakeDoc:
    def __init__(
        self,
        items: list[Any],
        *,
        page_count: int,
        tables: list[_FakeTable],
    ) -> None:
        self._items = items
        self._page_count = page_count
        self.tables = tables

    def iterate_items(self, page_no: int | None = None):
        for item in self._items:
            if page_no is None:
                yield item, 1
                continue
            prov = getattr(item, "prov", [])
            pages = {getattr(p, "page_no", None) for p in prov}
            if page_no in pages:
                yield item, 1

    def num_pages(self) -> int:
        return self._page_count

    def export_to_markdown(self, page_no: int | None = None) -> str:
        lines: list[str] = []
        for item, _level in self.iterate_items(page_no=page_no):
            if isinstance(item, SectionHeaderItem):
                lines.append(f"## {item.text}")
            elif isinstance(item, TextItem):
                lines.append(item.text)
        return "\n\n".join(lines)

    def extract_items_range(
        self,
        *,
        start: Any,
        end: Any,
        start_inclusive: bool,
        end_inclusive: bool,
        delete: bool,
    ) -> _FakeDoc:
        del start_inclusive, end_inclusive, delete
        start_idx = self._items.index(start)
        end_idx = self._items.index(end)
        return _FakeDoc(
            self._items[start_idx : end_idx + 1],
            page_count=self._page_count,
            tables=self.tables,
        )


def _build_doc() -> _FakeDoc:
    items: list[Any] = [
        SectionHeaderItem("Introduction", level=1, page_no=1),
        TextItem("Intro body", page_no=1),
        SectionHeaderItem("Methods", level=1, page_no=2),
        TextItem("Methods body", page_no=2),
    ]
    tables = [_FakeTable("| A | B |\n|---|---|\n| 1 | 2 |", page_no=2)]
    return _FakeDoc(items, page_count=2, tables=tables)


def test_store_builds_toc_from_cached_doc() -> None:
    """Stored document should produce TOC with doc_id and tool hint."""
    from openbb_pydantic_ai.pdf._graph import build_toc
    from openbb_pydantic_ai.pdf._store import DocumentStore

    store = DocumentStore()
    doc = _build_doc()
    cached = store.store("doc-1", doc, "test.pdf")  # type: ignore[arg-type]
    toc = build_toc(cached, "doc-1")

    assert "doc_id: doc-1" in toc
    assert "Sections:" in toc
    assert "Introduction" in toc
    assert "pdf_query" in toc
    assert store.get("doc-1") is not None


def test_store_resolves_source_alias_and_section_lookup() -> None:
    """Source aliases should resolve and section matching should work."""
    from openbb_pydantic_ai.pdf._query import find_section_node, read_section_markdown
    from openbb_pydantic_ai.pdf._store import DocumentStore

    store = DocumentStore()
    doc = _build_doc()
    source = "https://example.com/report.pdf"
    cached = store.store("doc-1", doc, "test.pdf", source=source)  # type: ignore[arg-type]

    by_source = store.get_by_source(source)
    assert by_source is not None
    assert by_source[0] == "doc-1"

    node = find_section_node(cached, "method")
    assert node is not None

    markdown, provenance = read_section_markdown(cached, node)
    assert "Methods" in markdown
    assert isinstance(provenance, list)


def test_register_document_source_alias() -> None:
    """Additional aliases should resolve to the same cached document."""
    from openbb_pydantic_ai.pdf._store import DocumentStore

    store = DocumentStore()
    doc = _build_doc()
    store.store("doc-1", doc, "test.pdf")  # type: ignore[arg-type]

    store.register_source("doc-1", "file-c0ebd61e-39fa-4382-b82f-4374ea96a30f")

    by_alias = store.get_by_source("c0ebd61e-39fa-4382-b82f-4374ea96a30f")
    assert by_alias is None

    by_full_alias = store.get_by_source("file-c0ebd61e-39fa-4382-b82f-4374ea96a30f")
    assert by_full_alias is not None
    assert by_full_alias[0] == "doc-1"


def test_store_fifo_eviction() -> None:
    """Store should evict oldest entries once max size is exceeded."""
    from openbb_pydantic_ai.pdf._store import DocumentStore

    store = DocumentStore(max_entries=1)
    doc = _build_doc()
    store.store("first-doc", doc, "first.pdf")  # type: ignore[arg-type]
    store.store("second-doc", doc, "second.pdf")  # type: ignore[arg-type]

    assert store.get("first-doc") is None
    assert store.get("second-doc") is not None
