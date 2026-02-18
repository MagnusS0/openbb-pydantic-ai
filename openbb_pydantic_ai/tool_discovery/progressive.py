"""Progressive-discovery tagging helpers.

This module provides a small public API for marking toolsets (or functions)
to be merged into OpenBB's progressive discovery wrapper by the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

_PROGRESSIVE_ATTR = "__openbb_progressive_config__"

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProgressiveConfig:
    """Configuration for progressive-discovery tagging.

    Parameters
    ----------
    group : str | None, default None
        Optional group id shown in progressive discovery instructions.
    description : str | None, default None
        Optional group description shown in progressive discovery instructions.
    """

    group: str | None = None
    description: str | None = None


def add_to_progressive(
    target: T,
    *,
    group: str | None = None,
    description: str | None = None,
) -> T:
    """Mark a toolset (or function) for progressive discovery.

    Parameters
    ----------
    target : T
        Object to mark, typically an ``AbstractToolset`` instance.
    group : str | None, default None
        Optional group id shown in progressive discovery instructions.
    description : str | None, default None
        Optional group description shown in progressive discovery instructions.

    Returns
    -------
    T
        The same object, enabling fluent usage.
    """
    setattr(target, _PROGRESSIVE_ATTR, ProgressiveConfig(group, description))
    return target


@overload
def progressive(
    target: T,
    *,
    group: str | None = None,
    description: str | None = None,
    toolset: Any | None = None,
) -> T: ...


@overload
def progressive(
    target: None = None,
    *,
    group: str | None = None,
    description: str | None = None,
    toolset: Any | None = None,
) -> Callable[[T], T]: ...


def progressive(
    target: T | None = None,
    *,
    group: str | None = None,
    description: str | None = None,
    toolset: Any | None = None,
) -> T | Callable[[T], T]:
    """Decorator/marker for progressive discovery.

    Parameters
    ----------
    target : T | None, default None
        Target object when used as a direct marker call, e.g.
        ``progressive(toolset, group="custom")``.
        When omitted, returns a decorator.
    group : str | None, default None
        Optional group id shown in progressive discovery instructions.
    description : str | None, default None
        Optional group description shown in progressive discovery instructions.
    toolset : Any | None, default None
        Optional containing toolset for function-level usage. When provided,
        the toolset is tagged alongside the decorated function.

    Returns
    -------
    T | Callable[[T], T]
        The tagged object when called directly, or a decorator when ``target``
        is omitted.

    Examples
    --------
    Tag a toolset directly:
    ``progressive(custom_toolset, group="custom_tools")``

    Tag via decorator:
    ``@progressive(toolset=custom_toolset, group="custom_tools")``
    """

    def _apply(obj: T) -> T:
        if toolset is not None:
            add_to_progressive(toolset, group=group, description=description)
        return add_to_progressive(obj, group=group, description=description)

    if target is None:
        return _apply
    return _apply(target)


def get_progressive_config(target: Any) -> ProgressiveConfig | None:
    """Return progressive-discovery config if a target was tagged.

    Parameters
    ----------
    target : Any
        Object that may have been tagged with progressive metadata.

    Returns
    -------
    ProgressiveConfig | None
        Tag configuration when present, otherwise ``None``.
    """
    cfg = getattr(target, _PROGRESSIVE_ATTR, None)
    if isinstance(cfg, ProgressiveConfig):
        return cfg
    return None
