#!/usr/bin/env bash
# Shard SAM3 detection across N GPUs (one detached tmux session per GPU).
# Each shard writes detections_shard<i>.json; merge with merge_detections.py.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FHS="${FHS:-$(nix build --no-link --print-out-paths "$REPO#packages.x86_64-linux.default")/bin/coral-fish-dev}"
NG=${NG:-4}
for i in $(seq 0 $((NG-1))); do
  tmux kill-session -t det_g$i 2>/dev/null
  tmux new-session -d -s det_g$i -c "$REPO" \
    "$FHS -c 'export HF_HOME=$REPO/models/hf_cache CUDA_VISIBLE_DEVICES=$i SHARD=$i NSHARDS=$NG; uv run python $HERE/full_pipeline_detect.py' > $HERE/detect_g$i.log 2>&1; echo DONE_g$i >> $HERE/detect_g$i.log"
  echo "launched det_g$i on GPU $i"
done
tmux ls 2>&1
