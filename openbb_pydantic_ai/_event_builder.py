"""Event builder utilities for consistent SSE event creation."""

from __future__ import annotations

from typing import Any, Literal

from openbb_ai.models import (
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)


class EventBuilder:
    """Provides consistent interface for creating SSE events."""

    @staticmethod
    def reasoning(
        message: str,
        *,
        details: dict[str, Any] | list[dict[str, Any] | str] | None = None,
        event_type: Literal["INFO", "WARNING", "ERROR"] = "INFO",
        artifacts: list[ClientArtifact] | None = None,
        hidden: bool = False,
    ) -> StatusUpdateSSE:
        """Create a reasoning/status update event.

        Parameters
        ----------
        message : str
            The reasoning message
        details : dict | list | None
            Optional details to include
        event_type : Literal["INFO", "WARNING", "ERROR"]
            The type of event
        artifacts : list[ClientArtifact] | None
            Optional artifacts to include
        hidden : bool
            Whether to hide this event

        Returns
        -------
        StatusUpdateSSE
            A status update SSE event
        """
        # Normalize dict details to list format expected by StatusUpdateSSEData
        normalized_details: list[dict[str, Any] | str] | None = None
        if details is not None:
            if isinstance(details, dict):
                normalized_details = [details]
            else:
                # details is already a list[dict[str, Any] | str]
                normalized_details = details  # type: ignore[assignment]

        return StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType=event_type,
                message=message,
                group="reasoning",
                details=normalized_details,
                artifacts=artifacts,
                hidden=hidden,
            )
        )

    @staticmethod
    def reasoning_with_artifacts(
        message: str,
        artifacts: list[ClientArtifact],
    ) -> StatusUpdateSSE:
        """Create a reasoning event with inline artifacts.

        Parameters
        ----------
        message : str
            The reasoning message
        artifacts : list[ClientArtifact]
            Artifacts to include inline

        Returns
        -------
        StatusUpdateSSE
            A status update SSE event with artifacts
        """
        return EventBuilder.reasoning(
            message,
            event_type="INFO",
            artifacts=artifacts,
        )

    @staticmethod
    def message(content: str) -> MessageChunkSSE:
        """Create a message chunk event.

        Parameters
        ----------
        content : str
            The message content

        Returns
        -------
        MessageChunkSSE
            A message chunk SSE event
        """
        return MessageChunkSSE(data=MessageChunkSSEData(delta=content))

    @staticmethod
    def artifact(artifact: ClientArtifact) -> MessageArtifactSSE:
        """Create an artifact event.

        Parameters
        ----------
        artifact : ClientArtifact
            The artifact to send

        Returns
        -------
        MessageArtifactSSE
            An artifact SSE event
        """
        return MessageArtifactSSE(data=artifact)

    @staticmethod
    def citations(citation_list: list[Citation]) -> CitationCollectionSSE:
        """Create a citation collection event.

        Parameters
        ----------
        citation_list : list[Citation]
            List of citations

        Returns
        -------
        CitationCollectionSSE
            A citation collection SSE event
        """
        return CitationCollectionSSE(data=CitationCollection(citations=citation_list))

    @staticmethod
    def error(
        message: str, *, details: dict[str, Any] | None = None
    ) -> StatusUpdateSSE:
        """Create an error event.

        Parameters
        ----------
        message : str
            The error message
        details : dict | None
            Optional error details

        Returns
        -------
        StatusUpdateSSE
            An error status update SSE event
        """
        return EventBuilder.reasoning(message, event_type="ERROR", details=details)

    @staticmethod
    def warning(
        message: str, *, details: dict[str, Any] | None = None
    ) -> StatusUpdateSSE:
        """Create a warning event.

        Parameters
        ----------
        message : str
            The warning message
        details : dict | None
            Optional warning details

        Returns
        -------
        StatusUpdateSSE
            A warning status update SSE event
        """
        return EventBuilder.reasoning(message, event_type="WARNING", details=details)
