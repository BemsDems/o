#!/usr/bin/env bash
set -euo pipefail

# Copies the multi_ticker_tcn_sonnet bundle (entrypoint + project/*) into clipboard.
# macOS: uses pbcopy
# Linux: tries xclip/xsel

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FILES=(
  "$ROOT_DIR/multi_ticker_tcn_sonnet.py"
  "$ROOT_DIR/project/__init__.py"
  "$ROOT_DIR/project/config.py"
  "$ROOT_DIR/project/main.py"
  "$ROOT_DIR/project/model.py"
  "$ROOT_DIR/project/data_loader.py"
  "$ROOT_DIR/project/sequences.py"
  "$ROOT_DIR/project/metrics.py"
  "$ROOT_DIR/project/diagnostics.py"
)

missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing file: $f" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

out=""
out+="# TCN bundle export\n"
out+="# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")\n"
out+="# Root: $ROOT_DIR\n\n"

for f in "${FILES[@]}"; do
  rel="${f#"$ROOT_DIR/"}"
  out+="\n---\n## $rel\n\n\`\`\`python\n"
  out+="$(cat "$f")\n"
  out+="\`\`\`\n"
done

copy_to_clipboard() {
  if command -v pbcopy >/dev/null 2>&1; then
    printf "%b" "$out" | pbcopy
    echo "Copied to clipboard via pbcopy. Size: $(printf "%b" "$out" | wc -c | tr -d ' ') bytes"
    return 0
  fi

  if command -v xclip >/dev/null 2>&1; then
    printf "%b" "$out" | xclip -selection clipboard
    echo "Copied to clipboard via xclip."
    return 0
  fi

  if command -v xsel >/dev/null 2>&1; then
    printf "%b" "$out" | xsel --clipboard --input
    echo "Copied to clipboard via xsel."
    return 0
  fi

  echo "No clipboard tool found (pbcopy/xclip/xsel). Printing to stdout instead." >&2
  printf "%b" "$out"
}

copy_to_clipboard
