#!/usr/bin/env bash
# Install the SAM3 package (cloned in external/sam3) + its undeclared deps into the venv.
# Requires the gated facebook/sam3 weights already fetched into models/hf_cache.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
cd "$REPO"
export HF_HOME="$PWD/models/hf_cache"
echo "=== installing SAM3 package + deps ==="
uv pip install -e external/sam3 einops pycocotools psutil 2>&1 | tail -40
echo "=== import test ==="
uv run python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import OK')"
echo "=== exit $? ==="
