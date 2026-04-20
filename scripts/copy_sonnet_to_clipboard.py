#!/usr/bin/env python3
from __future__ import annotations

"""Copy multi-ticker SONNET code (entrypoint + project modules) to macOS clipboard.

Usage:
  cd o-repo
  python3 scripts/copy_sonnet_to_clipboard.py

Result:
  - Full concatenated code is copied into clipboard via pbcopy.
"""

from pathlib import Path
import subprocess


def main() -> None:
    base = Path(__file__).resolve().parents[1]  # o-repo/

    files = [
        base / "multi_ticker_tcn_sonnet.py",
        base / "project" / "__init__.py",
        base / "project" / "config.py",
        base / "project" / "data_loader.py",
        base / "project" / "sequences.py",
        base / "project" / "model.py",
        base / "project" / "metrics.py",
        base / "project" / "diagnostics.py",
        base / "project" / "main.py",
    ]

    parts: list[str] = []
    parts.append("# AUTO-GENERATED: multi_ticker_tcn_sonnet (entrypoint + project modules)\n\n")

    for p in files:
        rel = p.relative_to(base)
        parts.append(f"### FILE: {rel}\n\n")
        if not p.exists():
            parts.append(f"# MISSING: {rel}\n\n")
            continue
        parts.append(p.read_text(encoding="utf-8").rstrip() + "\n\n")

    text = "".join(parts)

    # macOS clipboard
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    print(f"Copied to clipboard: {len(text)} chars")


if __name__ == "__main__":
    main()
