"""Tests for PDF query toolset actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inline_snapshot import snapshot
from pydantic_ai import ToolReturn

from openbb_pydantic_ai.pdf._graph import (
    CachedDocument,
    DocumentGraph,
    SectionInfo,
    SectionNode,
    TableInfo,
)
from openbb_pydantic_ai.pdf._toolsets import PdfAction, PdfQueryParams, _pdf_query
from tests.helpers.snapshot_utils import as_builtins


def _empty_cached_document() -> CachedDocument:
    graph = DocumentGraph(
        toc=(),
        nodes={},
        parent={},
        children={},
        next={},
        tables={},
        table_order=(),
    )
    return CachedDocument(
        doc=object(),  # type: ignore[arg-type]
        filename="report.pdf",
        page_count=10,
        sections=(),
        table_count=0,
        graph=graph,
    )


def test_pdf_query_returns_cache_miss_message(mocker) -> None:
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.get_document",
        return_value=None,
    )
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.get_document_by_source",
        return_value=None,
    )
    params = PdfQueryParams(doc_id="missing-doc", action=PdfAction.get_tables)

    result = _pdf_query(None, params)  # type: ignore[arg-type]
    assert isinstance(result, str)
    assert "not searchable yet" in result


def test_pdf_query_read_pages_emits_tool_return_with_citation(mocker) -> None:
    @dataclass
    class _BBox:
        l: float  # noqa: E741 - matches Docling bbox attribute names
        t: float
        r: float
        b: float

    @dataclass
    class _Prov:
        page_no: int
        bbox: _BBox

    cached = _empty_cached_document()
    provenance: list[Any] = [
        (_Prov(page_no=2, bbox=_BBox(1.0, 2.0, 3.0, 4.0)), "sample text")
    ]

    mocker.patch("openbb_pydantic_ai.pdf._toolsets.get_document", return_value=cached)
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.read_pages_markdown",
        return_value=("### Page 2\n\ncontent", provenance),
    )

    params = PdfQueryParams(
        doc_id="doc-123",
        action=PdfAction.read_pages,
        start_page=2,
        end_page=2,
    )
    result = _pdf_query(None, params)  # type: ignore[arg-type]

    assert isinstance(result, ToolReturn)
    assert result.return_value == "### Page 2\n\ncontent"
    assert isinstance(result.metadata, dict)
    citations = result.metadata.get("citations")
    assert isinstance(citations, list)
    assert len(citations) == 1
    assert citations[0].quote_bounding_boxes
    assert as_builtins(citations[0].quote_bounding_boxes[0]) == snapshot(
        [
            {
                "bottom": 4.0,
                "page": 2,
                "text": "sample text",
                "top": 2.0,
                "x0": 1.0,
                "x1": 3.0,
            }
        ]
    )


def test_pdf_query_resolves_alias_doc_id_for_citations(mocker) -> None:
    @dataclass
    class _BBox:
        l: float  # noqa: E741 - matches Docling bbox attribute names
        t: float
        r: float
        b: float

    @dataclass
    class _Prov:
        page_no: int
        bbox: _BBox

    cached = _empty_cached_document()
    provenance: list[Any] = [
        (_Prov(page_no=1, bbox=_BBox(0.1, 0.2, 0.3, 0.4)), "sample text")
    ]

    mocker.patch("openbb_pydantic_ai.pdf._toolsets.get_document", return_value=None)
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.get_document_by_source",
        return_value=("canonical-doc-id", cached),
    )
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.read_pages_markdown",
        return_value=("### Page 1\n\ncontent", provenance),
    )

    params = PdfQueryParams(
        doc_id="file-c0ebd61e-39fa-4382-b82f-4374ea96a30f",
        action=PdfAction.read_pages,
        start_page=1,
        end_page=1,
    )
    result = _pdf_query(None, params)  # type: ignore[arg-type]

    assert isinstance(result, ToolReturn)
    assert isinstance(result.metadata, dict)
    citations = result.metadata.get("citations")
    assert isinstance(citations, list)
    assert citations
    assert citations[0].source_info.metadata["doc_id"] == "canonical-doc-id"


def test_pdf_query_get_tables_formats_table_listing(mocker) -> None:
    cached = _empty_cached_document()
    mocker.patch("openbb_pydantic_ai.pdf._toolsets.get_document", return_value=cached)
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.list_tables",
        return_value=(
            TableInfo(
                index=0,
                table_id="table_0",
                page_no=5,
                caption="Revenue Table",
                preview="| col_a | col_b |",
            ),
        ),
    )

    params = PdfQueryParams(doc_id="doc-123", action=PdfAction.get_tables)
    result = _pdf_query(None, params)  # type: ignore[arg-type]

    assert isinstance(result, str)
    assert "[0]" in result
    assert "Revenue Table" in result
    assert "preview" in result


def test_pdf_query_read_section_by_index(mocker) -> None:
    @dataclass
    class _BBox:
        l: float  # noqa: E741 - matches Docling bbox attribute names
        t: float
        r: float
        b: float

    @dataclass
    class _Prov:
        page_no: int
        bbox: _BBox

    node = SectionNode(
        info=SectionInfo(
            index=1,
            section_id="section_1",
            heading="Methods",
            level=2,
            page_no=3,
        ),
        start_item_idx=10,
        end_item_idx=20,
    )
    cached = _empty_cached_document()

    mocker.patch("openbb_pydantic_ai.pdf._toolsets.get_document", return_value=cached)
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.get_section_node_by_index",
        return_value=node,
    )
    mocker.patch(
        "openbb_pydantic_ai.pdf._toolsets.read_section_markdown",
        return_value=(
            "## Methods\n\nBody",
            [(_Prov(page_no=3, bbox=_BBox(0, 0, 1, 1)), "Methods Body")],
        ),
    )

    params = PdfQueryParams(
        doc_id="doc-123",
        action=PdfAction.read_section,
        section_index=1,
    )
    result = _pdf_query(None, params)  # type: ignore[arg-type]

    assert isinstance(result, ToolReturn)
    assert "Methods" in result.return_value
    assert isinstance(result.metadata, dict)
    citations = result.metadata.get("citations")
    assert isinstance(citations, list)
    assert citations[0].source_info.type == "direct retrieval"
