"""PDF query toolset for selective document retrieval."""

from __future__ import annotations

from enum import Enum
from typing import Any

from openbb_ai.models import Citation, CitationHighlightBoundingBox, SourceInfo
from pydantic import BaseModel, Field, model_validator
from pydantic_ai import FunctionToolset, ToolReturn
from pydantic_ai.tools import RunContext

from openbb_pydantic_ai._config import PDF_QUERY_TOOL_NAME
from openbb_pydantic_ai._dependencies import OpenBBDeps
from openbb_pydantic_ai.pdf._citations import extract_citations_from_provenance
from openbb_pydantic_ai.pdf._graph import CachedDocument
from openbb_pydantic_ai.pdf._query import (
    find_section_node,
    get_section_node_by_index,
    list_tables,
    read_pages_markdown,
    read_section_markdown,
    read_table_markdown,
)
from openbb_pydantic_ai.pdf._store import get_document, get_document_by_source
from openbb_pydantic_ai.pdf._types import ProvenanceItem

_MAX_PAGE_WINDOW = 5
_MAX_HEADING_SUGGESTIONS = 8


class PdfAction(str, Enum):
    read_section = "read_section"
    read_pages = "read_pages"
    get_tables = "get_tables"
    read_table = "read_table"


class PdfQueryParams(BaseModel):
    """Parameters for ``pdf_query`` tool calls."""

    doc_id: str = Field(description="Document identifier from injected TOC prompt")
    action: PdfAction
    section: str | None = Field(
        default=None,
        description="Section heading text to retrieve",
    )
    section_index: int | None = Field(
        default=None,
        description="Section index from TOC to retrieve",
    )
    start_page: int | None = Field(
        default=None,
        description="Start page number (1-indexed)",
    )
    end_page: int | None = Field(
        default=None,
        description="End page number (1-indexed)",
    )
    table_index: int | None = Field(
        default=None,
        description="Table index from `get_tables` result",
    )

    @model_validator(mode="after")
    def validate_by_action(self) -> PdfQueryParams:
        if self.action == PdfAction.read_section:
            if self.section is None and self.section_index is None:
                raise ValueError("read_section requires `section` or `section_index`.")

        if self.action == PdfAction.read_pages:
            if self.start_page is None and self.end_page is None:
                raise ValueError("read_pages requires `start_page` and/or `end_page`.")

            # At least one is set (checked above); default the other to match
            start = self.start_page if self.start_page is not None else self.end_page
            end = self.end_page if self.end_page is not None else self.start_page
            if start is None or end is None:
                raise ValueError("read_pages requires `start_page` and/or `end_page`.")

            if start < 1 or end < 1:
                raise ValueError("Page numbers are 1-indexed and must be >= 1.")
            if end < start:
                raise ValueError(
                    "`end_page` must be greater than or equal to `start_page`."
                )
            if end - start + 1 > _MAX_PAGE_WINDOW:
                raise ValueError(
                    f"read_pages supports at most {_MAX_PAGE_WINDOW} pages per call."
                )

        if self.action == PdfAction.read_table and self.table_index is None:
            raise ValueError("read_table requires `table_index`.")

        return self


def _build_citation(
    *,
    cached: CachedDocument,
    doc_id: str,
    label: str,
    details: dict[str, Any],
    provenance_items: list[tuple[ProvenanceItem, str]],
) -> Citation:
    raw_boxes = extract_citations_from_provenance(provenance_items)
    boxes = [
        CitationHighlightBoundingBox(
            text=box.text,
            page=box.page,
            x0=box.x0,
            top=box.top,
            x1=box.x1,
            bottom=box.bottom,
        )
        for box in raw_boxes
    ]

    source = SourceInfo(
        type="direct retrieval",
        name=cached.filename,
        description=label,
        metadata={
            "doc_id": doc_id,
            "filename": cached.filename,
            **details,
        },
        citable=True,
    )
    return Citation(
        source_info=source,
        details=[details],
        quote_bounding_boxes=[boxes] if boxes else None,
    )


def _tool_return_with_citation(content: str, citation: Citation) -> ToolReturn:
    return ToolReturn(
        return_value=content,
        metadata={"citations": [citation]},
    )


def _section_not_found(cached: CachedDocument, target: str) -> str:
    hints = [section.heading for section in cached.sections[:_MAX_HEADING_SUGGESTIONS]]
    hint_text = ", ".join(hints) if hints else "no section headings available"
    return f"Section '{target}' was not found. Available headings include: {hint_text}"


def _read_section(cached: CachedDocument, params: PdfQueryParams) -> str | ToolReturn:
    node = None
    section_label = ""

    if params.section_index is not None:
        node = get_section_node_by_index(cached, params.section_index)
        section_label = str(params.section_index)

    if node is None and params.section:
        section_label = params.section
        if params.section.isdigit():
            node = get_section_node_by_index(cached, int(params.section))
        if node is None:
            node = find_section_node(cached, params.section)

    if node is None:
        return _section_not_found(cached, section_label)

    markdown, provenance = read_section_markdown(cached, node)
    if not markdown:
        return "Section content is empty."

    details = {
        "section_index": node.info.index,
        "section_id": node.info.section_id,
        "section_heading": node.info.heading,
    }
    if node.info.page_no is not None:
        details["page_no"] = node.info.page_no

    citation = _build_citation(
        cached=cached,
        doc_id=params.doc_id,
        label=f"PDF section: {node.info.heading}",
        details=details,
        provenance_items=provenance,
    )
    return _tool_return_with_citation(markdown, citation)


def _read_pages(cached: CachedDocument, params: PdfQueryParams) -> str | ToolReturn:
    # Guaranteed non-None by PdfQueryParams.validate_by_action
    start_page = params.start_page if params.start_page is not None else params.end_page
    end_page = params.end_page if params.end_page is not None else params.start_page
    if start_page is None or end_page is None:
        return "Missing page range parameters."

    if end_page > cached.page_count:
        return (
            f"Requested pages {start_page}-{end_page}, but document has "
            f"{cached.page_count} page(s)."
        )

    markdown, provenance = read_pages_markdown(cached, start_page, end_page)
    if not markdown:
        return "No extractable content found in that page range."

    details = {
        "start_page": start_page,
        "end_page": end_page,
    }
    citation = _build_citation(
        cached=cached,
        doc_id=params.doc_id,
        label=f"PDF pages {start_page}-{end_page}",
        details=details,
        provenance_items=provenance,
    )
    return _tool_return_with_citation(markdown, citation)


def _get_tables(cached: CachedDocument) -> str:
    tables = list_tables(cached)
    if not tables:
        return "No tables were detected in this document."

    lines = [f"Detected {len(tables)} table(s):", ""]
    for table in tables:
        page = f"p.{table.page_no}" if table.page_no is not None else "p.?"
        caption = table.caption or "(no caption)"
        lines.append(f"- [{table.index}] {page} | {caption} | preview: {table.preview}")
    return "\n".join(lines)


def _read_table(cached: CachedDocument, params: PdfQueryParams) -> str | ToolReturn:
    if params.table_index is None:
        return "read_table requires `table_index`."
    table = read_table_markdown(cached, params.table_index)
    if table is None:
        return (
            f"Table index {params.table_index} is out of range. "
            "Use action='get_tables' first."
        )

    markdown, provenance, table_info = table
    if not markdown:
        return f"Table {params.table_index} has no markdown output."

    heading = f"Table {table_info.index}"
    page = f"p.{table_info.page_no}" if table_info.page_no is not None else "p.?"
    content = f"### {heading} ({page})\n\n{markdown}"

    details = {
        "table_index": table_info.index,
        "table_id": table_info.table_id,
    }
    if table_info.page_no is not None:
        details["page_no"] = table_info.page_no

    citation = _build_citation(
        cached=cached,
        doc_id=params.doc_id,
        label=f"PDF table: {heading}",
        details=details,
        provenance_items=provenance,
    )
    return _tool_return_with_citation(content, citation)


def _pdf_query(ctx: RunContext[OpenBBDeps], params: PdfQueryParams) -> str | ToolReturn:
    """
    Use this tool to retrieve content from an uploaded PDF document
    You first need to call the widget tool with the PDF file to get the doc_id.

    With the doc_id, you can call this tool with different actions:
    - read_section: Retrieve a specific section by heading text or section index.
    - read_pages: Retrieve content from a range of pages.
    - get_tables: List detected tables in the document.
    - read_table: Retrieve a specific table by its index from the get_tables result.
    """
    del ctx

    doc_id = params.doc_id
    cached = get_document(doc_id)
    if cached is None:
        by_source = get_document_by_source(doc_id)
        if by_source is not None:
            doc_id, cached = by_source

    if cached is None:
        return (
            f"Document '{params.doc_id}' is not searchable yet. "
            "Call a widget tool that returns the PDF you want first, then use "
            "`pdf_query` with the doc_id from the TOC message."
        )

    if doc_id != params.doc_id:
        params = params.model_copy(update={"doc_id": doc_id})

    if params.action == PdfAction.read_section:
        return _read_section(cached, params)
    if params.action == PdfAction.read_pages:
        return _read_pages(cached, params)
    if params.action == PdfAction.get_tables:
        return _get_tables(cached)
    if params.action == PdfAction.read_table:
        return _read_table(cached, params)

    return f"Unsupported action: {params.action}"


def build_pdf_toolset() -> FunctionToolset[OpenBBDeps]:
    """Build PDF query toolset."""
    toolset = FunctionToolset[OpenBBDeps]()
    toolset.add_function(_pdf_query, name=PDF_QUERY_TOOL_NAME)
    return toolset
