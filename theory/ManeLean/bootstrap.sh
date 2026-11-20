#!/usr/bin/env bash
# Toolchain bootstrap for the ManeLean proofs: fetch deps at the pinned mathlib
# tag, sync the project toolchain to mathlib's, pull the prebuilt olean cache,
# and build the library.
set -uo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.elan/bin:$PATH"

echo "=== [1/4] lake update (resolve deps at pinned rev) ==="
lake update 2>&1 | tail -5
echo "=== [2/4] sync toolchain from mathlib ==="
if [ -f .lake/packages/mathlib/lean-toolchain ]; then
  cp .lake/packages/mathlib/lean-toolchain ./lean-toolchain
  echo "toolchain now: $(cat lean-toolchain)"
  lake update 2>&1 | tail -5
fi
echo "=== [3/4] lake exe cache get ==="
lake exe cache get 2>&1 | tail -6 || echo "CACHE GET FAILED (build will be slow)"
echo "=== [4/4] lake build ==="
lake build 2>&1 | tail -40
STATUS=$?
echo "=== bootstrap done, build exit=$STATUS ==="
exit $STATUS
