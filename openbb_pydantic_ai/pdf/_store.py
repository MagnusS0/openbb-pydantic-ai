"""Thread-safe in-memory cache for extracted PDF documents."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openbb_pydantic_ai.pdf._graph import CachedDocument, build_cached_document

if TYPE_CHECKING:
    from docling.datamodel.document import DoclingDocument


DEFAULT_CACHE_MAX_ENTRIES = 20
DEFAULT_CACHE_TTL_SECONDS = 60 * 60


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
        cached = build_cached_document(doc, filename)

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
            now = time.monotonic()
            self._prune_expired(now)
            entry = self._cache.get(doc_id)
            if entry is None or entry.expires_at <= now:
                if entry is not None:
                    self._cache.pop(doc_id, None)
                    self._cleanup_entry(doc_id, entry)
                self._source_map.pop(source_value, None)
                return None
            return doc_id, entry.cached

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
