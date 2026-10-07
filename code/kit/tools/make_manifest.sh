#!/usr/bin/env bash
# Regenerate MANIFEST.sha256 over every file of the kit (run after code/build/build_all.sh staged the .so files).
# apply_r16.sh refuses a kit whose site/ overlay/ launcher/ hold a file the manifest does not list, or whose bytes differ.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$KIT"
find . -type f ! -name MANIFEST.sha256 ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > MANIFEST.sha256
echo "MANIFEST.sha256: $(wc -l < MANIFEST.sha256) files"
