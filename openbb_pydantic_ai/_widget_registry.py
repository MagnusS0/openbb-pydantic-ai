"""Widget registry for centralized widget discovery and lookup."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

from openbb_ai.models import (
    LlmClientFunctionCallResultMessage,
    Widget,
    WidgetCollection,
)

from ._config import GET_WIDGET_DATA_TOOL_NAME

if TYPE_CHECKING:
    from pydantic_ai.toolsets import FunctionToolset

    from ._dependencies import OpenBBDeps


class WidgetRegistry:
    """Centralized registry for widget discovery and lookup."""

    def __init__(
        self,
        collection: WidgetCollection | None = None,
        toolsets: Sequence[FunctionToolset[OpenBBDeps]] | None = None,
    ):
        """Initialize widget registry from collection and toolsets.

        Parameters
        ----------
        collection : WidgetCollection | None
            Widget collection with priority groups
        toolsets : Sequence[FunctionToolset[OpenBBDeps]] | None
            Widget toolsets
        """
        self._by_tool_name: dict[str, Widget] = {}
        self._by_uuid: dict[str, Widget] = {}
        self._by_origin_id: dict[tuple[str, str], Widget] = {}

        def _register(widget: Widget, *, tool_name: str | None = None) -> None:
            if tool_name is not None:
                self._by_tool_name[tool_name] = widget
            uuid_key = str(widget.uuid)
            self._by_uuid.setdefault(uuid_key, widget)
            origin_key = (widget.origin, widget.widget_id)
            self._by_origin_id.setdefault(origin_key, widget)

        # Build lookup from toolsets
        if toolsets:
            for toolset in toolsets:
                widgets = getattr(toolset, "widgets_by_tool", None)
                if widgets:
                    for tool_name, widget in widgets.items():
                        _register(widget, tool_name=tool_name)

        # Also index from collection if provided
        if collection:
            for widget in self._iter_collection(collection):
                _register(widget)

    @staticmethod
    def _iter_collection(collection: WidgetCollection) -> Iterator[Widget]:
        """Iterate all widgets in a collection."""
        for group in (collection.primary, collection.secondary, collection.extra):
            yield from group

    def find_by_tool_name(self, name: str) -> Widget | None:
        """Find a widget by its tool name.

        Parameters
        ----------
        name : str
            The tool name to search for

        Returns
        -------
        Widget | None
            The widget if found, None otherwise
        """
        return self._by_tool_name.get(name)

    def find_by_uuid(self, uuid: str) -> Widget | None:
        """Find a widget by its UUID string.

        Parameters
        ----------
        uuid : str
            The UUID to search for

        Returns
        -------
        Widget | None
            The widget if found, None otherwise
        """
        return self._by_uuid.get(uuid)

    def find_by_origin_and_id(self, origin: str, widget_id: str) -> Widget | None:
        """Find a widget by its workspace origin/id pair."""

        return self._by_origin_id.get((origin, widget_id))

    def find_for_result(
        self, result: LlmClientFunctionCallResultMessage
    ) -> Widget | None:
        """Find the widget that produced a result message.

        Parameters
        ----------
        result : LlmClientFunctionCallResultMessage
            The result message to find a widget for

        Returns
        -------
        Widget | None
            The widget if found, None otherwise
        """
        # Check direct tool name match
        widget = self.find_by_tool_name(result.function)
        if widget is not None:
            return widget

        # Check if it's a get_widget_data call
        if result.function == GET_WIDGET_DATA_TOOL_NAME:
            data_sources = result.input_arguments.get("data_sources", [])
            if data_sources:
                widget_uuid = data_sources[0].get("widget_uuid")
                if widget_uuid:
                    return self.find_by_uuid(widget_uuid)

        return None

    def iter_all(self) -> Iterator[Widget]:
        """Iterate all registered widgets.

        Returns
        -------
        Iterator[Widget]
            Iterator over all widgets
        """
        # Use dict to deduplicate by UUID
        seen = set()
        for widget in self._by_uuid.values():
            if str(widget.uuid) not in seen:
                seen.add(str(widget.uuid))
                yield widget

    def as_mapping(self) -> Mapping[str, Widget]:
        """Get widget lookup as a read-only mapping by tool name.

        Returns
        -------
        Mapping[str, Widget]
            Read-only mapping from tool names to widgets
        """
        return self._by_tool_name
