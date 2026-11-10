"""
impenetra.garak_plugins.setup
=============================
Registers and integrates Impenetra's custom Garak probes and detectors into
the active Python environment's Garak installation.

Commands:
    python -m impenetra.garak_plugins.setup install
    python -m impenetra.garak_plugins.setup uninstall
    python -m impenetra.garak_plugins.setup status
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def _get_garak_dir() -> Path:
    try:
        import garak

        return Path(garak.__file__).parent
    except ImportError:
        print("❌ Error: garak is not installed in the active environment.", file=sys.stderr)
        sys.exit(1)


def _get_plugin_src_dir() -> Path:
    return Path(__file__).parent


def install_plugins() -> bool:
    """Install or symlink custom probes and detectors into garak's package."""
    garak_dir = _get_garak_dir()
    src_dir = _get_plugin_src_dir()

    probe_src = src_dir / "probes" / "off_charter.py"
    probe_dst = garak_dir / "probes" / "off_charter.py"

    detector_src = src_dir / "detectors" / "off_charter.py"
    detector_dst = garak_dir / "detectors" / "off_charter.py"

    for src, dst in [(probe_src, probe_dst), (detector_src, detector_dst)]:
        if not src.exists():
            print(f"❌ Source file not found: {src}", file=sys.stderr)
            return False

        if dst.is_symlink() or dst.exists():
            try:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
            except Exception as e:
                print(f"⚠️ Warning removing existing {dst}: {e}")

        try:
            os.symlink(src.resolve(), dst)
            print(f"🔗 Linked: {dst} -> {src}")
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
            print(f"📋 Copied: {src} -> {dst}")

    # Rebuild garak plugin cache
    _refresh_garak_cache()
    print("✅ Successfully registered off_charter probes and detectors with Garak!")
    return True


def uninstall_plugins() -> bool:
    """Remove custom probes and detectors from garak's package."""
    garak_dir = _get_garak_dir()
    for category in ["probes", "detectors"]:
        target = garak_dir / category / "off_charter.py"
        if target.is_symlink() or target.exists():
            target.unlink()
            print(f"🗑️ Removed: {target}")

    _refresh_garak_cache()
    print("✅ Uninstalled Impenetra plugins from Garak.")
    return True


def _refresh_garak_cache():
    """Trigger Garak's PluginCache rebuild."""
    try:
        from garak import _config
        from garak._plugins import PluginCache

        # Invalidate memory cache
        PluginCache._plugin_cache_dict = None

        # Remove user cache file if exists to force full rebuild
        user_cache = _config.transient.cache_dir / "resources" / "plugin_cache.json"
        if user_cache.exists():
            user_cache.unlink()

        # Rebuild
        PluginCache.instance()
    except Exception as e:
        print(f"ℹ️ Note: Cache refresh skipped or deferred: {e}")


def status():
    """Report whether plugins are registered."""
    garak_dir = _get_garak_dir()
    probe_dst = garak_dir / "probes" / "off_charter.py"
    detector_dst = garak_dir / "detectors" / "off_charter.py"

    print("Impenetra Garak Plugin Status:")
    print(f"  Garak Directory: {garak_dir}")
    print(f"  Probe off_charter.py:    {'Installed' if probe_dst.exists() else 'NOT installed'}")
    print(f"  Detector off_charter.py: {'Installed' if detector_dst.exists() else 'NOT installed'}")

    try:
        from garak._plugins import enumerate_plugins

        probes = [p for p, _ in enumerate_plugins("probes") if "off_charter" in p]
        detectors = [d for d, _ in enumerate_plugins("detectors") if "off_charter" in d]
        print(f"  Discovered probes:    {probes}")
        print(f"  Discovered detectors: {detectors}")
    except Exception as e:
        print(f"  Discovery check error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Impenetra Garak Plugin Setup")
    parser.add_argument(
        "action", choices=["install", "uninstall", "status"], default="status", nargs="?"
    )
    args = parser.parse_args()

    if args.action == "install":
        install_plugins()
    elif args.action == "uninstall":
        uninstall_plugins()
    elif args.action == "status":
        status()


if __name__ == "__main__":
    main()
