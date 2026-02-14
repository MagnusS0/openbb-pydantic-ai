"""Preprocessing of PDF content in tool result messages for LLM consumption.

Replaces raw PDF bytes/URLs in result messages with compact table-of-contents
summaries so the LLM can selectively query sections via ``pdf_query``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote

from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    LlmClientFunctionCallResultMessage,
    RawObjectDataFormat,
    SingleDataContent,
    SingleFileReference,
)

from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai.pdf._graph import build_toc
from openbb_pydantic_ai.pdf._store import (
    get_document,
    get_document_by_source,
    register_document_source,
    store_document,
)

logger = logging.getLogger(__name__)

# Per-document asyncio locks to prevent duplicate concurrent extractions.
# Keyed by document identity (source URL or content-hash doc_id).
_extraction_locks: dict[str, asyncio.Lock] = {}
_extraction_locks_guard = asyncio.Lock()


async def _get_extraction_lock(key: str) -> asyncio.Lock:
    """Return (or create) an asyncio.Lock for the given document key."""
    async with _extraction_locks_guard:
        lock = _extraction_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _extraction_locks[key] = lock
        return lock


# Type alias for data entry union
DataEntry = Any  # ClientCommandResult | DataContent | DataFileReferences | ...
PdfLikeItem = SingleDataContent | SingleFileReference

# Keys in data-source mappings that may contain document identifiers
_ID_KEYS = {
    "id",
    "uuid",
    "widget_id",
    "widget_uuid",
    "file_id",
    "file_uuid",
    "storedfileuuid",
}

_TEXT_DATA_FORMAT = RawObjectDataFormat(
    data_type="object",
    parse_as="text",
)


def _normalize_identifier(value: str) -> set[str]:
    """Generate all lookup forms for a single identifier string."""
    raw = value.strip()
    if not raw:
        return set()

    forms = {raw, unquote(raw)}

    if raw.startswith("file-") and len(raw) > 5:
        forms.add(raw[5:])

    if raw.startswith(("http://", "https://")):
        no_query = raw.split("?", 1)[0]
        forms.add(no_query)
        tail = unquote(no_query.rsplit("/", 1)[-1])
        if tail:
            forms.add(tail)

    forms.discard("")
    return forms


def _identifiers_from_mapping(mapping: Mapping[str, Any]) -> set[str]:
    """Extract identifiers from a mapping's id/file/uuid fields."""
    ids: set[str] = set()
    for key, value in mapping.items():
        if not isinstance(value, str):
            continue
        key_lower = key.lower()
        if (
            key_lower in _ID_KEYS
            or "file" in key_lower
            or "uuid" in key_lower
            or key_lower.endswith("_id")
        ):
            ids.update(_normalize_identifier(value))
    return ids


def _message_aliases_for_data_index(
    message: LlmClientFunctionCallResultMessage,
    data_index: int,
    data_count: int,
) -> set[str]:
    """Extract widget identifiers from ``input_arguments.data_sources``."""
    if message.function != GET_WIDGET_DATA_TOOL_NAME:
        return set()

    data_sources = message.input_arguments.get("data_sources")
    if not isinstance(data_sources, list) or not data_sources:
        return set()

    selected: list[dict[str, Any]] = []
    if data_count == 1:
        selected.extend(s for s in data_sources if isinstance(s, dict))
    elif data_index < len(data_sources) and isinstance(data_sources[data_index], dict):
        selected.append(data_sources[data_index])

    aliases: set[str] = set()
    for source in selected:
        aliases.update(_identifiers_from_mapping(source))
        input_args = source.get("input_args")
        if isinstance(input_args, Mapping):
            aliases.update(_identifiers_from_mapping(input_args))

    return aliases


def _is_pdf_mapping(mapping: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Check if a mapping describes a PDF and return its filename."""
    for container in (mapping.get("data_format"), mapping):
        if not isinstance(container, Mapping):
            continue
        data_type = container.get("data_type")
        if isinstance(data_type, str) and data_type.lower() == "pdf":
            filename = container.get("filename")
            if isinstance(filename, str) and filename.strip():
                return True, unquote(filename.strip())
            return True, None
    return False, None


def _find_pdf_sources(
    raw_content: str, default_filename: str
) -> list[tuple[str, str, set[str]]]:
    """Parse JSON content and return ``(source, filename, aliases)`` for each PDF found.

    Does a flat + one-level-deep scan (PDF metadata is never deeply nested
    in practice).
    """
    try:
        parsed = json.loads(raw_content)
    except (TypeError, ValueError):
        return []

    # Handle double-encoded JSON strings
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (TypeError, ValueError):
            return []

    results: dict[str, tuple[str, str, set[str]]] = {}

    nodes = parsed if isinstance(parsed, list) else [parsed]
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        _scan_node(node, default_filename, results)
        # One level deep into nested values
        for nested in node.values():
            if isinstance(nested, Mapping):
                _scan_node(nested, default_filename, results)
            elif isinstance(nested, list):
                for item in nested:
                    if isinstance(item, Mapping):
                        _scan_node(item, default_filename, results)

    return list(results.values())


def _scan_node(
    node: Mapping[str, Any],
    default_filename: str,
    results: dict[str, tuple[str, str, set[str]]],
) -> None:
    is_pdf, filename_override = _is_pdf_mapping(node)
    if not is_pdf:
        return

    filename = filename_override or default_filename
    aliases = _identifiers_from_mapping(node)
    if filename:
        aliases.update(_normalize_identifier(filename))

    for key in ("content", "url", "file_reference", "fileReference"):
        candidate = node.get(key)
        if isinstance(candidate, str) and candidate.strip():
            source = candidate.strip()
            aliases.update(_normalize_identifier(source))
            results[source] = (source, filename, aliases)


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _is_pdf_item(item: PdfLikeItem) -> bool:
    if item.data_format is None:
        return False
    return item.data_format.data_type == "pdf"


def _get_filename(item: PdfLikeItem) -> str:
    if item.data_format is None:
        return "document.pdf"
    return getattr(item.data_format, "filename", None) or "document.pdf"


def _doc_id_from_base64(content: str) -> str:
    pdf_bytes = base64.b64decode(content)
    return hashlib.sha256(pdf_bytes).hexdigest()


def _doc_id_for_source(source: str) -> str | None:
    if _is_url(source):
        hit = get_document_by_source(source)
        return hit[0] if hit else None

    try:
        doc_id = _doc_id_from_base64(source)
    except (ValueError, binascii.Error) as e:
        logger.warning("Failed to decode base64 content: %s", e)
        return None

    return doc_id if get_document(doc_id) is not None else None


async def _extract_pdf_toc(content: str, filename: str) -> str | None:
    """Extract/store PDF and return a compact TOC prompt.

    Uses per-document asyncio locks so that concurrent coroutines targeting the
    same document do not redundantly extract it (TOCTOU guard).
    """
    try:
        from openbb_pydantic_ai.pdf import extract_pdf_document
    except ImportError:
        logger.debug("PDF dependencies not installed, skipping extraction")
        return None

    try:
        if _is_url(content):
            # Determine a stable lock key from the source URL.
            lock = await _get_extraction_lock(content)
            async with lock:
                source_hit = get_document_by_source(content)
                if source_hit is not None:
                    doc_id, cached = source_hit
                    return build_toc(cached, doc_id)

                result, doc = await extract_pdf_document(
                    url=content,
                    filename=filename,
                )
                doc_id = str(result.metadata.get("doc_id") or "")
                if not doc_id:
                    msg = "PDF extraction did not produce doc_id"
                    raise RuntimeError(msg)

                cached = get_document(doc_id)
                if cached is None:
                    cached = store_document(doc_id, doc, filename, source=content)
                return build_toc(cached, doc_id)

        expected_doc_id = _doc_id_from_base64(content)
        lock = await _get_extraction_lock(expected_doc_id)
        async with lock:
            cached = get_document(expected_doc_id)
            if cached is not None:
                return build_toc(cached, expected_doc_id)

            result, doc = await extract_pdf_document(
                content=content,
                filename=filename,
            )
            doc_id = str(result.metadata.get("doc_id") or expected_doc_id)
            cached = get_document(doc_id)
            if cached is None:
                cached = store_document(doc_id, doc, filename)
            return build_toc(cached, doc_id)
    except Exception as e:
        logger.warning("Failed to extract PDF TOC: %s", e)
        return None


async def _process_pdf_source(
    source: str,
    filename: str,
    extra_aliases: set[str],
) -> str | None:
    """Extract a PDF, register all aliases, return TOC text or None."""
    toc = await _extract_pdf_toc(source, filename)
    if not toc:
        return None

    doc_id = _doc_id_for_source(source)
    if doc_id:
        all_aliases = (
            extra_aliases
            | _normalize_identifier(source)
            | _normalize_identifier(filename)
        )
        for alias in all_aliases:
            register_document_source(doc_id, alias)

    return toc


def _toc_item(content: str, *, citable: bool = True) -> SingleDataContent:
    return SingleDataContent(
        content=content,
        data_format=_TEXT_DATA_FORMAT,
        citable=citable,
    )


def _pdf_fallback(filename: str) -> SingleDataContent:
    return SingleDataContent(
        content=f"[PDF '{filename}' could not be extracted]",
        data_format=_TEXT_DATA_FORMAT,
        citable=False,
    )


async def _process_content_item(
    item: SingleDataContent,
    message_aliases: set[str],
) -> tuple[SingleDataContent, bool]:
    """Process a single content item, replacing PDF with TOC if applicable.

    Returns (item, was_modified).
    """
    filename = _get_filename(item)

    # Direct PDF item (data_format.data_type == "pdf")
    if _is_pdf_item(item):
        toc = await _process_pdf_source(item.content, filename, message_aliases)
        if toc:
            return _toc_item(toc, citable=item.citable), True
        return _pdf_fallback(filename), True

    # JSON-embedded PDF metadata
    pdf_sources = _find_pdf_sources(item.content, filename)
    if not pdf_sources:
        return item, False

    tocs: list[str] = []
    fallback_filename = filename
    for source, fname, aliases in pdf_sources:
        fallback_filename = fname
        all_aliases = message_aliases | aliases
        toc = await _process_pdf_source(source, fname, all_aliases)
        if toc:
            tocs.append(toc)

    if not tocs:
        return _pdf_fallback(fallback_filename), True
    return _toc_item("\n\n---\n\n".join(tocs), citable=item.citable), True


async def _process_file_ref_item(
    item: SingleFileReference,
    message_aliases: set[str],
) -> SingleDataContent | None:
    """Process a PDF file reference, returning TOC content or None for non-PDFs."""
    if not _is_pdf_item(item):
        return None

    filename = _get_filename(item)
    source = str(item.url)
    toc = await _process_pdf_source(source, filename, message_aliases)
    if toc:
        return _toc_item(toc, citable=item.citable)
    return _pdf_fallback(filename)


async def preprocess_pdf_in_result(
    message: LlmClientFunctionCallResultMessage,
) -> LlmClientFunctionCallResultMessage:
    """Replace PDF bytes/URLs in a result message with TOC prompts."""
    if not message.data:
        return message

    modified = False
    new_data: list[DataEntry] = []

    for data_index, data_entry in enumerate(message.data):
        message_aliases = _message_aliases_for_data_index(
            message,
            data_index,
            len(message.data),
        )

        # -- DataContent: process each item in place --
        if isinstance(data_entry, DataContent) and data_entry.items:
            new_items: list[SingleDataContent] = []
            for item in data_entry.items:
                processed, changed = await _process_content_item(item, message_aliases)
                if changed:
                    modified = True
                new_items.append(processed)

            new_data.append(data_entry.model_copy(update={"items": new_items}))
            continue

        # -- DataFileReferences: convert PDF refs to content items --
        if isinstance(data_entry, DataFileReferences) and data_entry.items:
            has_pdf = any(_is_pdf_item(item) for item in data_entry.items)
            if not has_pdf:
                new_data.append(data_entry)
                continue

            modified = True
            converted: list[SingleDataContent] = []
            for item in data_entry.items:
                pdf_item = await _process_file_ref_item(item, message_aliases)
                if pdf_item is not None:
                    converted.append(pdf_item)
                else:
                    # Non-PDF reference: keep as text
                    converted.append(
                        SingleDataContent(
                            content=str(item.url),
                            data_format=_TEXT_DATA_FORMAT,
                            citable=getattr(item, "citable", True),
                        )
                    )

            new_data.append(
                DataContent(
                    items=converted,
                    extra_citations=data_entry.extra_citations,
                )
            )
            continue

        new_data.append(data_entry)

    if not modified:
        return message

    return message.model_copy(update={"data": new_data})


async def preprocess_pdf_in_results(
    results: list[LlmClientFunctionCallResultMessage],
) -> list[LlmClientFunctionCallResultMessage]:
    """Pre-process multiple result messages, replacing PDF payloads with TOC."""
    return list(await asyncio.gather(*(preprocess_pdf_in_result(r) for r in results)))
