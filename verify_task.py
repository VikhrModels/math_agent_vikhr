#!/usr/bin/env python3
"""
verify_task.py

A minimal helper script to *verify* a Lean 4 task (a `.lean` file) by compiling it with `lake`.

Usage
-----
    python verify_task.py /absolute/path/to/task.lean

The script will:
1. Ensure the provided file exists (expects an **absolute** path).
2. Ascend from the file location until a directory containing a `lakefile.lean` or
   `lake-manifest.json` is found – this directory is treated as the Lake project root.
3. Run:
       /home/ismail/.elan/bin/lake env lean --make <file>
   in that root directory.
4. Forward compiler output to the terminal and exit with the same status code as Lean.

If verification succeeds (exit-code 0) the script prints a ✓ message, otherwise it
prints ✗ and returns the non-zero exit-code.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
import shutil

def resolve_lake_binary() -> str:
    """Resolve path to `lake` binary with sensible fallbacks.

    Priority:
    1) `LAKE_BINARY` environment variable if it points to an existing file
    2) `lake` discovered on PATH via `shutil.which`
    3) Default `$HOME/.elan/bin/lake`
    """
    env_binary = os.getenv("LAKE_BINARY")
    if env_binary and Path(env_binary).exists():
        return env_binary
    which_binary = shutil.which("lake")
    if which_binary:
        return which_binary
    return str(Path.home() / ".elan" / "bin" / "lake")

# Path to Lake binary (can be overridden via LAKE_BINARY env)
LAKE_BINARY = resolve_lake_binary()

# Always use the miniF2F-lean4 directory as the Lake project root
PROJECT_ROOT = Path(__file__).resolve().parent / "miniF2F-lean4"

# Sanity-check project root at import time so errors surface early
if not (PROJECT_ROOT / "lake-manifest.json").exists() and not (PROJECT_ROOT / "lakefile.lean").exists():
    print(f"[verify_task] Error: Expected Lake project root at {PROJECT_ROOT} but none found (no lake-manifest.json or lakefile.lean).",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Verification helper
# ---------------------------------------------------------------------------

# Retain fallback in case someone changes PROJECT_ROOT, but default is constant.
def find_lake_project_root(file_path: Path) -> Path | None:
    """Walk up the directory tree looking for a Lake project root (helper)."""
    for parent in [file_path] + list(file_path.parents):
        if (parent / "lakefile.lean").exists() or (parent / "lake-manifest.json").exists():
            return parent
    return None


def verify_lean_file(lean_file: Path) -> int:
    """Compile `lean_file` with Lake using the fixed project root; return exit code."""
    print(f"[verify_task] Lake project root: {PROJECT_ROOT}")

    if not Path(LAKE_BINARY).exists():
        print(
            "[verify_task] Error: Could not locate `lake` executable. Set LAKE_BINARY env variable or install elan (which provides lake).\n"
            f"  Tried: {LAKE_BINARY}",
            file=sys.stderr,
        )
        return 1

    cmd = [LAKE_BINARY, "env", "lean", str(lean_file)]
    print(f"[verify_task] Running: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

    # Forward compiler output
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)

    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a Lean 4 task (Lean code) by compiling it with Lake.\n\n"
                    "The Lean code can be provided in three ways (checked in this order):\n"
                    "  1. --file <path> : Path to a file containing Lean code.\n"
                    "  2. --code <string> : Lean code passed as a CLI string.  Be sure to quote it.\n"
                    "  3. No arguments : Lean code is read from STDIN.  Finish with Ctrl-D."
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--file", dest="code_file", help="Absolute path to a file with Lean code.")
    group.add_argument("--code", dest="code_string", help="Lean code passed directly as a string.")
    args = parser.parse_args()

    # --- Obtain Lean code ---------------------------------------------------
    if args.code_file:
        lean_path = Path(args.code_file).expanduser().resolve()
        if not lean_path.exists():
            print(f"[verify_task] Error: File not found – {lean_path}", file=sys.stderr)
            sys.exit(1)
        with lean_path.open("r", encoding="utf-8") as f:
            lean_code = f.read()
    elif args.code_string:
        lean_code = args.code_string
    else:
        print("[verify_task] Reading Lean code from STDIN.  Press Ctrl-D to finish.\n")
        lean_code = sys.stdin.read()

    lean_code = lean_code.lstrip()
    if not lean_code.strip():
        print("[verify_task] Error: Empty Lean code provided.", file=sys.stderr)
        sys.exit(1)

    # Ensure required import
    if "import" not in lean_code.splitlines()[0]:
        if "MiniF2F.Minif2fImport" not in lean_code:
            lean_code = "import MiniF2F.Minif2fImport\n\n" + lean_code

    # --- Write to temporary file -------------------------------------------
    import tempfile
    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".lean", dir=PROJECT_ROOT, delete=False, encoding="utf-8")
    tmp_file.write(lean_code)
    tmp_file.flush()
    tmp_file_path = Path(tmp_file.name)
    tmp_file.close()

    try:
        exit_code = verify_lean_file(tmp_file_path)
    finally:
        # Clean up the temporary file regardless of the outcome
        try:
            tmp_file_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[verify_task] Warning: Could not delete temp file {tmp_file_path}: {e}", file=sys.stderr)

    if exit_code == 0:
        print("\n[verify_task] ✓ Verification succeeded.")
    else:
        print("\n[verify_task] ✗ Verification failed.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
