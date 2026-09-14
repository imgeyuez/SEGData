#!/usr/bin/env bash
# Extract media/image.emf from each *_Aufgabe.docx under VeraAufgaben
# and write alongside as <basename>.emf

#Fail if an error occurs, if an undefined variable is used, or if a command in a pipeline fails
set -euo pipefail

# Root directory to scan (default: VeraAufgaben)
ROOT_DIR="${1:-VeraAufgaben}"

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: unzip is required but not installed." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: directory '$ROOT_DIR' not found." >&2
  exit 1
fi

export LC_ALL=C

# Find each DOCX that ends with _Aufgabe.docx and extract media/image.emf
find "$ROOT_DIR" -type f -name '*_Aufgabe.docx' -print0 |
while IFS= read -r -d '' docx; do
  base="${docx%.docx}"
  out="${base}.emf"

  if unzip -p "$docx" 'media/image.emf' > "$out" 2>/dev/null; then
    echo "Extracted media/image.emf -> $out"
  else
    echo "Warning: media/image.emf not found in $docx" >&2
    rm -f "$out" 2>/dev/null || true
  fi
done

