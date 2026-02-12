"""
Smoke test: verify that the built package installs and reports the correct version.
"""

import importlib
import importlib.metadata
import sys


def main():
    module_name = "openbb_pydantic_ai"
    dist_name = module_name.replace("_", "-")

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

    try:
        expected_version = importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ Installed distribution metadata not found for {dist_name}.")
        sys.exit(1)

    if str(found_version) != expected_version:
        print(
            f"❌ Version mismatch: module reports {found_version}, "
            f"but installed package metadata says {expected_version}"
        )
        sys.exit(1)

    print(f"✅ Version matches ({found_version})")
    print("🎉 Smoke test passed — distribution looks healthy.")


if __name__ == "__main__":
    main()
