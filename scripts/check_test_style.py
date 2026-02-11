#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_UNIT_DIR = ROOT / "tests" / "unit"


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _check_no_local_imports(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if line.startswith((" ", "\t")):
            stripped = line.lstrip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                rel = path.relative_to(ROOT)
                errors.append(
                    (
                        f"{rel}:{lineno}: local import found in indented scope; "
                        "keep imports at module level"
                    )
                )
    return errors


def _check_no_anyio_markers(path: Path, content: str) -> list[str]:
    errors: list[str] = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if "mark.anyio" in line:
            rel = path.relative_to(ROOT)
            errors.append(
                (
                    f"{rel}:{lineno}: found 'mark.anyio'; async tests should "
                    "rely on asyncio_mode=auto"
                )
            )
    return errors


def main() -> int:
    if not TESTS_UNIT_DIR.exists():
        print("tests/unit not found; skipping check.")
        return 0

    violations: list[str] = []

    for path in _iter_python_files(TESTS_UNIT_DIR):
        content = path.read_text(encoding="utf-8")
        violations.extend(_check_no_local_imports(path, content))
        violations.extend(_check_no_anyio_markers(path, content))

    if violations:
        print("Test style violations found:")
        for issue in violations:
            print(f"- {issue}")
        return 1

    print("Test style checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
