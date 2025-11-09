"""
Smoke test: verify that the built package installs and reports the correct version.
"""

import importlib
import sys
from pathlib import Path

import tomllib


def read_pyproject_info():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    name = data.get("project", {}).get("name")
    version = data.get("project", {}).get("version")
    if not (name and version):
        raise SystemExit(
            "Missing [project].name or [project].version in pyproject.toml"
        )
    return name, version


def main():
    pkg_name, expected_version = read_pyproject_info()
    module_name = pkg_name.replace("-", "_")

    print(f"📦 Importing {module_name} ...")
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"❌ Failed to import {module_name}: {e}")
        sys.exit(1)

    print(f"✅ Imported {module_name}")

    # Validate version
    found_version = getattr(mod, "__version__", None)
    if found_version is None:
        print("⚠️  No __version__ found in module — expected one.")
        sys.exit(1)

    if str(found_version) != str(expected_version):
        print(
            f"❌ Version mismatch: module reports {found_version}, "
            f"but pyproject.toml says {expected_version}"
        )
        sys.exit(1)

    print(f"✅ Version matches ({found_version})")
    print("🎉 Smoke test passed — distribution looks healthy.")


if __name__ == "__main__":
    main()
