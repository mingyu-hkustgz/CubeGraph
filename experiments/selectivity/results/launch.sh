#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd/ymy/CubeGraph/new-host-20260919/selectivity-20261028
test ! -e queue.pid
sha256sum -c source-6b3972e.tar.sha256
mkdir code
tar -xf source-6b3972e.tar -C code
printf '%s\n' 6b3972e > source-commit.txt
nohup bash code/experiments/selectivity/run.sh "$PWD" >queue.log 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >queue.pid
sleep 1
kill -0 "$pid"
cat queue.pid
head -n 8 queue.log
