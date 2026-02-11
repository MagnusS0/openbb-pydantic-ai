"""Document graph types, index building, and TOC formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openbb_pydantic_ai.pdf._types import DocItem, TableItem

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument


_HEADING_COLLAPSE_RE = re.compile(r"\s+")
_MAX_TABLE_PREVIEW_CHARS = 80


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SectionInfo:
    """Public section metadata for TOC display and lookup."""

    index: int
    section_id: str
    heading: str
    level: int
    page_no: int | None


@dataclass(slots=True, frozen=True)
class SectionNode:
    """Internal section node including item-range boundaries."""

    info: SectionInfo
    start_item_idx: int
    end_item_idx: int


@dataclass(slots=True, frozen=True)
class TableInfo:
    """Table metadata used for table listing and retrieval."""

    index: int
    table_id: str
    page_no: int | None
    caption: str | None
    preview: str


@dataclass(slots=True, frozen=True)
class DocumentGraph:
    """Structured index over sections/tables for progressive disclosure."""

    toc: tuple[str, ...]
    nodes: dict[str, SectionNode]
    parent: dict[str, str | None]
    children: dict[str, tuple[str, ...]]
    next: dict[str, str | None]
    tables: dict[str, TableInfo]
    table_order: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class CachedDocument:
    """Cached document payload with retrieval metadata."""

    doc: DoclingDocument
    filename: str
    page_count: int
    sections: tuple[SectionInfo, ...]
    table_count: int
    graph: DocumentGraph


# ---------------------------------------------------------------------------
# Item introspection helpers
# ---------------------------------------------------------------------------


def _page_no_from_item(item: DocItem) -> int | None:
    prov_list = getattr(item, "prov", None)
    if not prov_list:
        return None

    for prov in prov_list:
        page_no = getattr(prov, "page_no", None)
        if isinstance(page_no, int):
            return page_no
    return None


def normalize_heading(value: str) -> str:
    lowered = value.strip().lower()
    return _HEADING_COLLAPSE_RE.sub(" ", lowered)


def _is_section_item(item: DocItem) -> bool:
    cls_name = type(item).__name__
    return cls_name in ("SectionHeaderItem", "TitleItem")


def _heading_from_item(item: DocItem) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    orig = getattr(item, "orig", None)
    if isinstance(orig, str) and orig.strip():
        return orig.strip()

    return "Untitled Section"


def _level_from_item(item: DocItem) -> int:
    value = getattr(item, "level", 1)
    try:
        level = int(value)
    except (TypeError, ValueError):
        level = 1
    return max(1, level)


def collect_items(doc: DoclingDocument) -> tuple[Any, ...]:
    """Return all items from a ``DoclingDocument`` as a flat tuple."""
    return tuple(item for item, _level in doc.iterate_items())


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def _build_sections(items: tuple[Any, ...]) -> list[SectionNode]:
    headers: list[tuple[int, Any]] = []
    for idx, item in enumerate(items):
        if _is_section_item(item):
            headers.append((idx, item))

    if not headers:
        if not items:
            return []
        node = SectionNode(
            info=SectionInfo(
                index=0,
                section_id="section_0",
                heading="Document",
                level=1,
                page_no=_page_no_from_item(items[0]),
            ),
            start_item_idx=0,
            end_item_idx=len(items),
        )
        return [node]

    sections: list[SectionNode] = []
    for sec_idx, (item_idx, header_item) in enumerate(headers):
        heading = _heading_from_item(header_item)
        level = _level_from_item(header_item)
        page_no = _page_no_from_item(header_item)
        section_id = f"section_{sec_idx}"

        # End range extends to the next header at same-or-higher level
        end_idx = len(items)
        for next_idx, next_header_item in headers[sec_idx + 1 :]:
            if _level_from_item(next_header_item) <= level:
                end_idx = next_idx
                break

        sections.append(
            SectionNode(
                info=SectionInfo(
                    index=sec_idx,
                    section_id=section_id,
                    heading=heading,
                    level=level,
                    page_no=page_no,
                ),
                start_item_idx=item_idx,
                end_item_idx=end_idx,
            )
        )

    return sections


def _table_preview(markdown: str) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:_MAX_TABLE_PREVIEW_CHARS]
    return "(empty table)"


def _table_caption(item: TableItem) -> str | None:
    captions = getattr(item, "captions", None) or []
    for caption in captions:
        text = getattr(caption, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _build_tables(
    doc: DoclingDocument,
) -> tuple[dict[str, TableInfo], tuple[str, ...]]:
    tables: dict[str, TableInfo] = {}
    order: list[str] = []

    for idx, item in enumerate(getattr(doc, "tables", ())):
        table_id = f"table_{idx}"
        page_no = _page_no_from_item(item)
        caption = _table_caption(item)

        try:
            markdown = item.export_to_markdown(doc=doc).strip()
        except Exception:
            markdown = ""

        tables[table_id] = TableInfo(
            index=idx,
            table_id=table_id,
            page_no=page_no,
            caption=caption,
            preview=_table_preview(markdown),
        )
        order.append(table_id)

    return tables, tuple(order)


def _build_graph(
    doc: DoclingDocument,
) -> tuple[tuple[SectionInfo, ...], DocumentGraph]:
    items = collect_items(doc)
    section_nodes = _build_sections(items)

    nodes: dict[str, SectionNode] = {}
    toc: list[str] = []
    parent: dict[str, str | None] = {}
    children_map: dict[str, list[str]] = {}
    next_map: dict[str, str | None] = {}

    stack: list[SectionInfo] = []
    infos: list[SectionInfo] = []

    for idx, node in enumerate(section_nodes):
        info = node.info
        nodes[info.section_id] = node
        toc.append(info.section_id)
        infos.append(info)

        while stack and stack[-1].level >= info.level:
            stack.pop()
        parent_id = stack[-1].section_id if stack else None
        parent[info.section_id] = parent_id
        children_map.setdefault(info.section_id, [])
        if parent_id is not None:
            children_map.setdefault(parent_id, []).append(info.section_id)

        next_id = (
            section_nodes[idx + 1].info.section_id
            if idx + 1 < len(section_nodes)
            else None
        )
        next_map[info.section_id] = next_id
        stack.append(info)

    children = {key: tuple(value) for key, value in children_map.items()}
    tables, table_order = _build_tables(doc)

    graph = DocumentGraph(
        toc=tuple(toc),
        nodes=nodes,
        parent=parent,
        children=children,
        next=next_map,
        tables=tables,
        table_order=table_order,
    )

    return tuple(infos), graph


def build_cached_document(doc: DoclingDocument, filename: str) -> CachedDocument:
    """Build a ``CachedDocument`` with full section/table graph from a docling doc."""
    sections, graph = _build_graph(doc)
    return CachedDocument(
        doc=doc,
        filename=filename,
        page_count=doc.num_pages(),
        sections=sections,
        table_count=len(graph.tables),
        graph=graph,
    )


# ---------------------------------------------------------------------------
# TOC formatting
# ---------------------------------------------------------------------------


def build_toc(cached: CachedDocument, key: str) -> str:
    """Build a compact TOC summary for model-context injection."""
    lines: list[str] = [
        "Document has been extracted, here is the table of contents:",
        "",
        f'PDF: "{cached.filename}" (doc_id: {key})',
        f"Pages: {cached.page_count} | Tables: {cached.table_count}",
        "",
        "Sections:",
    ]

    if not cached.sections:
        lines.append("(No section headers detected)")
    else:
        for section in cached.sections:
            page = f"p.{section.page_no}" if section.page_no is not None else "p.?"
            lines.append(
                f"{section.index}. [H{section.level}] {section.heading} ({page})"
            )

    lines.extend(
        [
            "",
            f'Use the `pdf_query` tool with doc_id="{key}"'
            " to read sections, pages, or tables.",
        ]
    )

    return "\n".join(lines)
