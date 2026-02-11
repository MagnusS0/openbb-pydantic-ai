from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

_adapter = TypeAdapter(object)


def as_builtins(value: Any) -> Any:
    """Recursively convert dataclasses and models to plain Python builtins."""
    return _adapter.dump_python(value)
