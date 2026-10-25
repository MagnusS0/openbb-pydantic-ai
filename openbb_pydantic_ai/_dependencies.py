"""Dependency injection container for OpenBB workspace context."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from openbb_ai.models import (
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    LlmMessage,
    QueryRequest,
    RawContext,
    Widget,
    WidgetCollection,
    WorkspaceState,
)

from openbb_pydantic_ai._config import CONTEXT_SNAPSHOT_EXTRA_STATE_KEY
from openbb_pydantic_ai._utils import iter_widget_collection

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenBBDeps:
    """Dependency container passed to Pydantic AI runs.

    The dependency bundle exposes OpenBB Workspace specific context so that
    system prompts, tools, and output validators can access widget metadata or
    other request scoped information via ``RunContext[OpenBBDeps]``.

    Attributes:
        widgets: Collection of available widgets organized by priority
        context: Workspace context data (datasets, documents, etc.)
        urls: Relevant URLs for the current request
        workspace_state: Current workspace state including dashboard info
        timezone: User's timezone (defaults to UTC)
        state: Serialized workspace state as dictionary
    """

    widgets: WidgetCollection | None = None
    context: list[RawContext] | None = None
    urls: list[str] | None = None
    workspace_state: WorkspaceState | None = None
    timezone: str = "UTC"
    state: dict[str, Any] = field(default_factory=dict)

    def iter_widgets(self) -> Iterable[Widget]:
        """Yield all widgets across priority groups (primary, secondary, extra)."""
        if not self.widgets:
            return
        yield from iter_widget_collection(self.widgets)

    def get_widget_by_uuid(self, widget_uuid: str) -> Widget | None:
        """Find a widget by its UUID string."""
        for widget in self.iter_widgets():
            if str(widget.uuid) == widget_uuid:
                return widget
        return None


def build_deps_from_request(request: QueryRequest) -> OpenBBDeps:
    """Create an OpenBBDeps instance from an incoming QueryRequest."""
    ws = request.workspace_state
    return OpenBBDeps(
        widgets=request.widgets,
        context=list(request.context) if request.context is not None else None,
        urls=list(request.urls) if request.urls is not None else None,
        workspace_state=ws,
        timezone=request.timezone,
        state=ws.model_dump(exclude_none=True) if ws is not None else {},
    )


def build_context_snapshot(request: QueryRequest) -> dict[str, Any]:
    """Compact request context to round-trip via a function call's `extra_state`.

    The Workspace omits timezone, urls and workspace_state from the follow-up
    request that carries a function call result. Only the dashboard info is kept
    from the workspace state since it is all the instructions render.
    """
    snapshot: dict[str, Any] = {"timezone": request.timezone}
    if request.urls:
        snapshot["urls"] = list(request.urls)
    ws = request.workspace_state
    if ws is not None and ws.current_dashboard_info is not None:
        snapshot["workspace_state"] = {
            "current_dashboard_info": ws.current_dashboard_info.model_dump(
                mode="json", exclude_none=True
            )
        }
    return snapshot


def _trailing_chain_snapshot(messages: Sequence[LlmMessage]) -> dict[str, Any] | None:
    """Return the snapshot on the trailing result of the current chain.

    Only a final result that directly follows its call qualifies, so snapshots
    from earlier chains or from late results after assistant text are never used.
    """
    if len(messages) < 2:
        return None
    call, result = messages[-2], messages[-1]
    if not (
        isinstance(result, LlmClientFunctionCallResultMessage)
        and isinstance(call, LlmClientMessage)
        and isinstance(call.content, LlmClientFunctionCall)
        and call.content.function == result.function
    ):
        return None
    snapshot = (result.extra_state or {}).get(CONTEXT_SNAPSHOT_EXTRA_STATE_KEY)
    return snapshot if isinstance(snapshot, dict) else None


def restore_request_context(request: QueryRequest) -> QueryRequest:
    """Fill timezone, urls and workspace_state a follow-up request omitted.

    Values present on the request always win.
    """
    snapshot = _trailing_chain_snapshot(request.messages)
    if snapshot is None:
        return request

    updates: dict[str, Any] = {}

    timezone = snapshot.get("timezone")
    if "timezone" not in request.model_fields_set and isinstance(timezone, str):
        try:
            ZoneInfo(timezone)
        except Exception:  # noqa: BLE001
            logger.debug("Ignoring invalid timezone in context snapshot")
        else:
            updates["timezone"] = timezone

    urls = snapshot.get("urls")
    if (
        request.urls is None
        and isinstance(urls, list)
        and all(isinstance(url, str) for url in urls)
    ):
        updates["urls"] = urls

    ws = snapshot.get("workspace_state")
    if request.workspace_state is None and isinstance(ws, dict):
        try:
            updates["workspace_state"] = WorkspaceState.model_validate(ws)
        except ValueError:
            logger.debug("Ignoring invalid workspace_state in context snapshot")

    return request.model_copy(update=updates) if updates else request
