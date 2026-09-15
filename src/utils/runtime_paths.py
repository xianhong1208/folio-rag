
"""Runtime path resolution for dev / Nuitka / Docker deployments.

`Path(__file__).parent / "xxx"` works in dev but breaks under Nuitka onefile
mode: onefile extracts into a temp dir, so `__file__` resolves resources to
the stale build-time snapshot instead of the real config next to the binary
or the Docker volume mount. These helpers resolve external dirs uniformly
across onefile, standalone, Docker, and dev modes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional


def resolve_external_dir(name: str, dev_root: Path) -> Optional[Path]:
    """Locate an external resource directory across deployment modes.

    Returns the first candidate directory that exists, or None. Candidate
    order (highest to lowest priority): NUITKA_ONEFILE_PARENT's exe dir,
    /proc/self/exe dir, sys.executable dir, /app/{name}, dev_root/name.
    /app/{name} precedes dev_root so a Docker volume mount wins over the
    onefile bundled copy.
    """
    candidates: List[Path] = []

    # (1) Nuitka onefile: NUITKA_ONEFILE_PARENT is the PID of the parent
    # process that launched the onefile binary
    nuitka_parent = os.environ.get("NUITKA_ONEFILE_PARENT")
    if nuitka_parent:
        try:
            parent_exe = Path(f"/proc/{nuitka_parent}/exe").resolve()
            if parent_exe.exists():
                candidates.append(parent_exe.parent / name)
        except (OSError, ValueError):
            pass

    # (2) Linux-generic: /proc/self/exe always points to the binary the
    # current process is really executing. Reject the fake /tmp/onefile_*
    # path (the binary inside the onefile extraction dir)
    if sys.platform.startswith("linux"):
        try:
            proc_exe = Path("/proc/self/exe").resolve()
            if not str(proc_exe).startswith("/tmp/onefile_"):
                candidates.append(proc_exe.parent / name)
        except (OSError, ValueError):
            pass

    # (3) Nuitka standalone (non-onefile): sys.executable is the real binary
    try:
        exe_path = Path(sys.executable).resolve()
        if not str(exe_path).startswith("/tmp/onefile_"):
            candidates.append(exe_path.parent / name)
    except (OSError, ValueError):
        pass

    # (4) Hardcoded Docker container path. Deliberately placed before
    # dev_root so a volume mount's real contents win over the bundled copy
    candidates.append(Path(f"/app/{name}"))

    # (5) Dev mode / onefile bundled fallback
    candidates.append(dev_root / name)

    for p in candidates:
        if p.is_dir():
            return p
    return None


def resolve_base_dir(dev_root: Path) -> Path:
    """Locate the writable base directory across deployment modes.

    Unlike resolve_external_dir, the target need not already exist (suited to
    a writable output dir created on first deploy). Always returns a Path
    (final fallback dev_root). The binary-dir candidates are tried only in
    compiled-binary mode: under plain Python /proc/self/exe is the system
    Python bin dir, which must not be treated as the base.
    """
    main_mod = sys.modules.get("__main__")
    is_compiled = (
        getattr(sys, "frozen", False)  # PyInstaller
        or (main_mod is not None and hasattr(main_mod, "__compiled__"))  # Nuitka
        or "NUITKA_ONEFILE_PARENT" in os.environ  # Nuitka onefile redundant signal
    )

    if is_compiled:
        # (1) Nuitka onefile parent process
        nuitka_parent = os.environ.get("NUITKA_ONEFILE_PARENT")
        if nuitka_parent:
            try:
                parent_exe = Path(f"/proc/{nuitka_parent}/exe").resolve()
                if parent_exe.exists():
                    return parent_exe.parent
            except (OSError, ValueError):
                pass

        # (2) Linux /proc/self/exe
        if sys.platform.startswith("linux"):
            try:
                proc_exe = Path("/proc/self/exe").resolve()
                if not str(proc_exe).startswith("/tmp/onefile_"):
                    return proc_exe.parent
            except (OSError, ValueError):
                pass

        # (3) sys.executable
        try:
            exe_path = Path(sys.executable).resolve()
            if not str(exe_path).startswith("/tmp/onefile_"):
                return exe_path.parent
        except (OSError, ValueError):
            pass

    # (4) Dev fallback
    return dev_root
