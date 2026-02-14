"""Widget registry for centralized widget discovery and lookup."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from openbb_ai.models import (
    LlmClientFunctionCallResultMessage,
    Widget,
    WidgetCollection,
)

from openbb_pydantic_ai._config import GET_WIDGET_DATA_TOOL_NAME
from openbb_pydantic_ai._utils import get_first_data_source, iter_widget_collection

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

    from openbb_pydantic_ai._dependencies import OpenBBDeps


@runtime_checkable
class HasWidgetsByTool(Protocol):
    widgets_by_tool: Mapping[str, Widget]


class WidgetRegistry:
    """Centralized registry for widget discovery and lookup."""

    def __init__(
        self,
        collection: WidgetCollection | None = None,
        toolsets: Sequence[AbstractToolset[OpenBBDeps]] | None = None,
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

        def _register(widget: Widget, *, tool_name: str | None = None) -> None:
            if tool_name is not None:
                self._by_tool_name[tool_name] = widget
            uuid_key = str(widget.uuid)
            self._by_uuid.setdefault(uuid_key, widget)

        # Build lookup from toolsets
        if toolsets:
            for toolset in toolsets:
                if isinstance(toolset, HasWidgetsByTool):
                    for tool_name, widget in toolset.widgets_by_tool.items():
                        _register(widget, tool_name=tool_name)

        # Also index from collection if provided
        if collection:
            for widget in iter_widget_collection(collection):
                _register(widget)

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
            first_data_source = get_first_data_source(result.input_arguments)
            if first_data_source is None:
                return None

            widget_uuid = first_data_source.get("widget_uuid")
            if isinstance(widget_uuid, str):
                return self.find_by_uuid(widget_uuid)

        return None

    def iter_all(self) -> Iterator[Widget]:
        """Iterate all registered widgets.

        Returns
        -------
        Iterator[Widget]
            Iterator over all widgets
        """
        yield from self._by_uuid.values()

    def as_mapping(self) -> Mapping[str, Widget]:
        """Get widget lookup as a read-only mapping by tool name.

        Returns
        -------
        Mapping[str, Widget]
            Read-only mapping from tool names to widgets
        """
        return self._by_tool_name
