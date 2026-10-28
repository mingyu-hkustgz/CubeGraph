#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd/ymy/CubeGraph/new-host-20260919/shape-modes-20261027
trap 'rc=$?; printf "exit=%s time=%s\n" "$rc" "$(date -Is)" > queue.state' EXIT
echo "RESUME_YFCC_GEOMETRY_V2 $(date -Is)"
bash code-yfcc-v2/experiments/aspect_ratio/run.sh /mnt/ssd/ymy/CubeGraph/DATA yfcc /mnt/ssd/ymy/CubeGraph/new-host-20260919/bottom-up-20261015-final/indexes/yfcc_real_2d.cube "$PWD/yfcc" /mnt/ssd/ymy/CubeGraph/new-host-20260919/logs/runner.lock
