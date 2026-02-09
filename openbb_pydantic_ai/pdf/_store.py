"""In-memory PDF document store for TOC and selective retrieval."""

from __future__ import annotations

import difflib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument


DEFAULT_CACHE_MAX_ENTRIES = 20
DEFAULT_CACHE_TTL_SECONDS = 60 * 60
_HEADING_COLLAPSE_RE = re.compile(r"\s+")
_MAX_TABLE_PREVIEW_CHARS = 80


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


def _page_no_from_item(item: Any) -> int | None:
    prov_list = getattr(item, "prov", None)
    if not prov_list:
        return None

    for prov in prov_list:
        page_no = getattr(prov, "page_no", None)
        if isinstance(page_no, int):
            return page_no
    return None


def _normalize_heading(value: str) -> str:
    lowered = value.strip().lower()
    return _HEADING_COLLAPSE_RE.sub(" ", lowered)


def _is_section_item(item: Any) -> bool:
    cls_name = type(item).__name__
    return cls_name in ("SectionHeaderItem", "TitleItem")


def _heading_from_item(item: Any) -> str:
    text = getattr(item, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    orig = getattr(item, "orig", None)
    if isinstance(orig, str) and orig.strip():
        return orig.strip()

    return "Untitled Section"


def _level_from_item(item: Any) -> int:
    value = getattr(item, "level", 1)
    try:
        level = int(value)
    except (TypeError, ValueError):
        level = 1
    return max(1, level)


def _collect_items(doc: DoclingDocument) -> tuple[Any, ...]:
    return tuple(item for item, _level in doc.iterate_items())


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


def _table_caption(item: Any) -> str | None:
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
    items = _collect_items(doc)
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


def _build_cached_document(doc: DoclingDocument, filename: str) -> CachedDocument:
    sections, graph = _build_graph(doc)
    return CachedDocument(
        doc=doc,
        filename=filename,
        page_count=doc.num_pages(),
        sections=sections,
        table_count=len(graph.tables),
        graph=graph,
    )


@dataclass(slots=True)
class _CacheEntry:
    cached: CachedDocument
    expires_at: float
    sources: set[str]


class DocumentStore:
    """Thread-safe in-memory cache for extracted PDF documents."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_CACHE_MAX_ENTRIES,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._source_map: dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_entries = max(1, int(max_entries))
        self._ttl = max(1, int(ttl_seconds))

    def clear(self) -> None:
        """Remove all entries (useful in tests)."""
        with self._lock:
            self._cache.clear()
            self._source_map.clear()

    def store(
        self,
        key: str,
        doc: DoclingDocument,
        filename: str,
        *,
        source: str | None = None,
    ) -> CachedDocument:
        """Store a document and return the cached payload."""
        now = time.monotonic()
        cached = _build_cached_document(doc, filename)

        with self._lock:
            self._prune_expired(now)

            existing = self._cache.pop(key, None)
            if existing is not None:
                self._cleanup_entry(key, existing)

            sources: set[str] = set()
            if source:
                source_value = source.strip()
                if source_value:
                    sources.add(source_value)
                    self._source_map[source_value] = key

            self._cache[key] = _CacheEntry(
                cached=cached,
                expires_at=now + self._ttl,
                sources=sources,
            )

            while len(self._cache) > self._max_entries:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._cleanup_entry(evicted_key, evicted_entry)

        return cached

    def register_source(self, key: str, source: str) -> None:
        """Register an additional source alias for a cached document."""
        source_value = source.strip()
        if not source_value:
            return

        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            entry = self._cache.get(key)
            if entry is None:
                return
            entry.sources.add(source_value)
            self._source_map[source_value] = key

    def get(self, key: str) -> CachedDocument | None:
        """Return cached document by doc_id, or ``None``."""
        now = time.monotonic()
        with self._lock:
            self._prune_expired(now)
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._cache.pop(key, None)
                self._cleanup_entry(key, entry)
                return None
            return entry.cached

    def get_by_source(self, source: str) -> tuple[str, CachedDocument] | None:
        """Resolve a cached document by source alias."""
        source_value = source.strip()
        if not source_value:
            return None

        with self._lock:
            doc_id = self._source_map.get(source_value)
        if doc_id is None:
            return None

        cached = self.get(doc_id)
        if cached is None:
            with self._lock:
                self._source_map.pop(source_value, None)
            return None
        return doc_id, cached

    def _cleanup_entry(self, doc_id: str, entry: _CacheEntry) -> None:
        for source in entry.sources:
            if self._source_map.get(source) == doc_id:
                self._source_map.pop(source, None)

    def _prune_expired(self, now: float) -> None:
        expired = [
            doc_id for doc_id, entry in self._cache.items() if entry.expires_at <= now
        ]
        for doc_id in expired:
            entry = self._cache.pop(doc_id, None)
            if entry is not None:
                self._cleanup_entry(doc_id, entry)


_default_store = DocumentStore()


def configure_cache(*, max_entries: int, ttl_seconds: int) -> None:
    """Reconfigure the default store limits (mainly for tests)."""
    global _default_store
    _default_store = DocumentStore(max_entries=max_entries, ttl_seconds=ttl_seconds)


def clear_store() -> None:
    """Clear the default store."""
    _default_store.clear()


def store_document(
    key: str,
    doc: DoclingDocument,
    filename: str,
    *,
    source: str | None = None,
) -> CachedDocument:
    """Store a document in the default cache."""
    return _default_store.store(key, doc, filename, source=source)


def register_document_source(key: str, source: str) -> None:
    """Register a source alias in the default cache."""
    _default_store.register_source(key, source)


def get_document(key: str) -> CachedDocument | None:
    """Retrieve from the default cache by doc_id."""
    return _default_store.get(key)


def get_document_by_source(
    source: str,
) -> tuple[str, CachedDocument] | None:
    """Resolve from the default cache by source alias."""
    return _default_store.get_by_source(source)


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
    normalized_query = _normalize_heading(query)
    if not normalized_query:
        return None

    by_id: dict[str, str] = {}
    for section in cached.sections:
        normalized = _normalize_heading(section.heading)
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


def collect_provenance(doc: DoclingDocument) -> list[tuple[Any, str]]:
    """Collect ``(provenance_item, parent_text)`` pairs from a document.

    Docling's ``ProvenanceItem`` carries bbox/page but NOT the source text;
    the text lives on the parent ``DocItem``.  We pair them here so that
    downstream citation builders have access to both.
    """
    results: list[tuple[Any, str]] = []
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", "") or ""
        prov_list = getattr(item, "prov", None)
        if prov_list:
            for prov in prov_list:
                results.append((prov, text))
    return results


def read_section_markdown(
    cached: CachedDocument, node: SectionNode
) -> tuple[str, list[Any]]:
    """Extract markdown + provenance for a section node range."""
    items = _collect_items(cached.doc)
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
) -> tuple[str, list[tuple[Any, str]]]:
    """Extract markdown + provenance for a page range."""
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
) -> tuple[str, list[tuple[Any, str]], TableInfo] | None:
    """Extract markdown + provenance for a table index."""
    if table_index < 0 or table_index >= cached.table_count:
        return None

    table_id = cached.graph.table_order[table_index]
    table_info = cached.graph.tables[table_id]
    table_item = cached.doc.tables[table_index]
    markdown = table_item.export_to_markdown(doc=cached.doc).strip()
    text = getattr(table_item, "text", "") or ""
    prov_list = getattr(table_item, "prov", None) or []
    provenance = [(prov, text) for prov in prov_list]
    return markdown, provenance, table_info
