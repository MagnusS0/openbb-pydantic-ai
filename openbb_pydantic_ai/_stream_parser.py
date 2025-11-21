"""Stream parser for handling text with artifact placeholders."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from openbb_ai.helpers import message_chunk
from openbb_ai.models import SSE, MessageArtifactSSE, MessageChunkSSE

from openbb_pydantic_ai._config import CHART_PLACEHOLDER_TOKENS


@dataclass
class StreamParser:
    """Parses text streams to handle artifact placeholders."""

    tokens: tuple[str, ...] = field(default=CHART_PLACEHOLDER_TOKENS)
    _placeholder_buffer: str = field(default="")

    def parse(
        self,
        text: str,
        artifact_provider: Iterator[MessageArtifactSSE],
        on_text_streamed: Callable[[], None] | None = None,
    ) -> list[SSE]:
        """Parse text for artifact placeholders and emit events.

        Parameters
        ----------
        text : str
            The text chunk to parse.
        artifact_provider : Iterator[MessageArtifactSSE]
            Iterator that yields artifacts when a placeholder is encountered.
        on_text_streamed : callable[[], None] | None
            Callback to invoke when text is successfully streamed.

        Returns
        -------
        list[SSE]
            A list of SSE events (MessageChunkSSE or MessageArtifactSSE).
        """
        events: list[SSE] = []
        if not text and not self._placeholder_buffer:
            return events

        combined = f"{self._placeholder_buffer}{text}"
        self._placeholder_buffer = ""
        idx = 0

        while True:
            match = self._next_token(combined, idx)
            if match is None:
                break
            next_idx, token = match

            segment = combined[idx:next_idx]
            if segment:
                events.append(self._message_chunk(segment, on_text_streamed))

            try:
                artifact = next(artifact_provider)
                events.append(artifact)
                idx = next_idx + len(token)
            except StopIteration:
                # No artifact available yet for this token.
                # Stop parsing and buffer the remaining text (starting with the token)
                # so we can try again when more artifacts or text arrive.
                idx = next_idx
                break

        remaining = combined[idx:]
        if remaining:
            if self._starts_with_token(remaining):
                self._placeholder_buffer = remaining
            else:
                suffix_len = self._placeholder_suffix_len(remaining)
                emit_len = len(remaining) - suffix_len
                if emit_len > 0:
                    events.append(
                        self._message_chunk(remaining[:emit_len], on_text_streamed)
                    )
                if suffix_len > 0:
                    self._placeholder_buffer = remaining[-suffix_len:]

        return events

    def flush(self, on_text_streamed: Callable[[], None] | None = None) -> list[SSE]:
        """Flush any remaining buffered text."""
        events = []
        if self._placeholder_buffer:
            events.append(
                self._message_chunk(self._placeholder_buffer, on_text_streamed)
            )
            self._placeholder_buffer = ""
        return events

    def _message_chunk(
        self, content: str, on_text_streamed: Callable[[], None] | None
    ) -> MessageChunkSSE:
        if on_text_streamed:
            on_text_streamed()
        return message_chunk(content)

    def _next_token(self, text: str, start_idx: int) -> tuple[int, str] | None:
        earliest_idx = -1
        selected_token: str | None = None
        for token in self.tokens:
            token_idx = text.find(token, start_idx)
            if token_idx == -1:
                continue
            if earliest_idx == -1 or token_idx < earliest_idx:
                earliest_idx = token_idx
                selected_token = token
        if selected_token is None:
            return None
        return earliest_idx, selected_token

    def _starts_with_token(self, text: str) -> bool:
        return any(text.startswith(token) for token in self.tokens)

    def _placeholder_suffix_len(self, text: str) -> int:
        max_suffix = 0
        for token in self.tokens:
            max_len = min(len(text), len(token) - 1)
            for length in range(max_len, 0, -1):
                if token.startswith(text[-length:]):
                    max_suffix = max(max_suffix, length)
                    break
        return max_suffix

    def has_pending_placeholder(self) -> bool:
        """Return True when a placeholder token is buffered awaiting an artifact."""
        return any(token in self._placeholder_buffer for token in self.tokens)
