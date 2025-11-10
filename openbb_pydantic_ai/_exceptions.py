"""Custom exceptions for OpenBB Pydantic AI adapter."""

from __future__ import annotations


class OpenBBPydanticAIError(Exception):
    """Base exception for OpenBB Pydantic AI adapter errors."""


class WidgetNotFoundError(OpenBBPydanticAIError):
    """Raised when a widget cannot be found by tool name or UUID."""

    def __init__(self, identifier: str, lookup_type: str = "tool_name"):
        """Initialize with widget identifier and lookup type.

        Parameters
        ----------
        identifier : str
            The widget identifier that was not found
        lookup_type : str
            The type of lookup performed (tool_name, uuid, etc.)
        """
        super().__init__(f"Widget not found by {lookup_type}: {identifier}")
        self.identifier = identifier
        self.lookup_type = lookup_type


class InvalidToolCallError(OpenBBPydanticAIError):
    """Raised when a tool call is malformed or invalid."""

    def __init__(self, tool_name: str, reason: str):
        """Initialize with tool name and reason.

        Parameters
        ----------
        tool_name : str
            The name of the tool that had an invalid call
        reason : str
            The reason why the call is invalid
        """
        super().__init__(f"Invalid tool call for '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason


class SerializationError(OpenBBPydanticAIError):
    """Raised when content serialization or deserialization fails."""

    def __init__(self, content_type: str, reason: str):
        """Initialize with content type and reason.

        Parameters
        ----------
        content_type : str
            The type of content being serialized
        reason : str
            The reason for the failure
        """
        super().__init__(f"Serialization failed for {content_type}: {reason}")
        self.content_type = content_type
        self.reason = reason
