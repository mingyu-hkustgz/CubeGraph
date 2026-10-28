#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd/ymy/CubeGraph/new-host-20260919/shape-modes-20261027
trap 'rc=$?; printf "exit=%s time=%s\n" "$rc" "$(date -Is)" > queue.state' EXIT
BASE=/mnt/ssd/ymy/CubeGraph
ROOT="$BASE/new-host-20260919"
OLD="$ROOT/bottom-up-20261015-final"
for cfg in 'sift uniform_2d' 'yfcc real_2d'; do
 read -r DS META <<< "$cfg"
 bash code/experiments/aspect_ratio/run.sh "$BASE/DATA" "$DS" \
   "$OLD/indexes/${DS}_${META}.cube" "$PWD/$DS" "$ROOT/logs/runner.lock"
done
echo "ALL_COMPLETE $(date -Is)"
