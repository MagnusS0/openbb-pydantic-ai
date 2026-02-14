"""Section, page, and table retrieval from cached PDF documents."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from openbb_pydantic_ai.pdf._graph import (
    CachedDocument,
    SectionNode,
    TableInfo,
    collect_items,
    normalize_heading,
)
from openbb_pydantic_ai.pdf._types import ProvenanceItem

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument


def section_ids(cached: CachedDocument) -> tuple[str, ...]:
    """Return section identifiers in TOC order."""
    return cached.graph.toc


def get_section_node(cached: CachedDocument, section_id: str) -> SectionNode | None:
    """Return a section node by section identifier."""
    return cached.graph.nodes.get(section_id)


def get_section_node_by_index(
    cached: CachedDocument, section_index: int
) -> SectionNode | None:
    """Return a section node by TOC index."""
    if section_index < 0 or section_index >= len(cached.sections):
        return None
    section_id = cached.sections[section_index].section_id
    return cached.graph.nodes.get(section_id)


def find_section_node(cached: CachedDocument, query: str) -> SectionNode | None:
    """Find best matching section by heading text."""
    normalized_query = normalize_heading(query)
    if not normalized_query:
        return None

    by_id: dict[str, str] = {}
    for section in cached.sections:
        normalized = normalize_heading(section.heading)
        by_id[section.section_id] = normalized

        if normalized == normalized_query:
            return cached.graph.nodes.get(section.section_id)

    for section in cached.sections:
        normalized = by_id.get(section.section_id, "")
        if normalized_query in normalized or normalized in normalized_query:
            return cached.graph.nodes.get(section.section_id)

    keys = list(by_id.values())
    matches = difflib.get_close_matches(normalized_query, keys, n=1, cutoff=0.55)
    if not matches:
        return None

    best = matches[0]
    for section in cached.sections:
        if by_id.get(section.section_id) == best:
            return cached.graph.nodes.get(section.section_id)
    return None


def collect_provenance(doc: DoclingDocument) -> list[tuple[ProvenanceItem, str]]:
    """Collect ``(provenance_item, parent_text)`` pairs from a document.

    Docling's ``ProvenanceItem`` carries bbox/page but NOT the source text;
    the text lives on the parent ``DocItem``.  We pair them here so that
    downstream citation builders have access to both.
    """
    return [
        (prov, getattr(item, "text", "") or "")
        for item, _level in doc.iterate_items()
        for prov in (getattr(item, "prov", None) or ())
    ]


def read_section_markdown(
    cached: CachedDocument, node: SectionNode
) -> tuple[str, list[tuple[ProvenanceItem, str]]]:
    """Extract markdown + provenance for a section node range."""
    items = collect_items(cached.doc)
    if not items:
        return "", []

    start_idx = max(0, min(node.start_item_idx, len(items) - 1))
    end_idx = max(start_idx + 1, min(node.end_item_idx, len(items)))

    sub_doc = cached.doc.extract_items_range(
        start=items[start_idx],
        end=items[end_idx - 1],
        start_inclusive=True,
        end_inclusive=True,
        delete=False,
    )
    markdown = sub_doc.export_to_markdown().strip()
    provenance = collect_provenance(sub_doc)
    return markdown, provenance


def read_pages_markdown(
    cached: CachedDocument, start_page: int, end_page: int
) -> tuple[str, list[tuple[ProvenanceItem, str]]]:
    """Extract markdown + provenance for a page range."""
    start_page = max(1, min(start_page, cached.page_count))
    end_page = max(start_page, min(end_page, cached.page_count))

    sections: list[str] = []
    provenance: list[tuple[Any, str]] = []

    for page_no in range(start_page, end_page + 1):
        page_markdown = cached.doc.export_to_markdown(page_no=page_no).strip()
        if page_markdown:
            sections.append(f"### Page {page_no}\n\n{page_markdown}")

        for item, _level in cached.doc.iterate_items(page_no=page_no):
            text = getattr(item, "text", "") or ""
            prov_list = getattr(item, "prov", None)
            if prov_list:
                for prov in prov_list:
                    provenance.append((prov, text))

    return "\n\n".join(sections).strip(), provenance


def list_tables(cached: CachedDocument) -> tuple[TableInfo, ...]:
    """Return table metadata in index order."""
    return tuple(cached.graph.tables[table_id] for table_id in cached.graph.table_order)


def read_table_markdown(
    cached: CachedDocument, table_index: int
) -> tuple[str, list[tuple[ProvenanceItem, str]], TableInfo] | None:
    """Extract markdown + provenance for a table index."""
    if table_index < 0 or table_index >= cached.table_count:
        return None

    table_id = cached.graph.table_order[table_index]
    table_info = cached.graph.tables[table_id]
    table_item = cached.doc.tables[table_info.index]
    markdown = table_item.export_to_markdown(doc=cached.doc).strip()
    text = getattr(table_item, "text", "") or ""
    prov_list = getattr(table_item, "prov", None) or []
    provenance = [(prov, text) for prov in prov_list]
    return markdown, provenance, table_info
