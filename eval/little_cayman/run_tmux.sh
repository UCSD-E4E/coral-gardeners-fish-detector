#!/usr/bin/env bash
# Launch a command inside the project's Nix FHS sandbox in a detached tmux session.
# Survives SSH disconnect. Usage: run_tmux.sh <session_name> <logfile> <command...>
set -u
SESSION="$1"; LOG="$2"; shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
# Resolve the FHS wrapper binary from the flake (override by exporting FHS=...).
FHS="${FHS:-$(nix build --no-link --print-out-paths "$REPO#packages.x86_64-linux.default")/bin/coral-fish-dev}"
CMD="$*"
tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "$FHS -c '$CMD' > '$LOG' 2>&1; echo \"[[EXIT \$?]]\" >> '$LOG'"
echo "launched tmux session '$SESSION' -> $LOG"
tmux ls 2>/dev/null
