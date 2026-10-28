#!/usr/bin/env bash
set -euo pipefail
cd /mnt/ssd/ymy/CubeGraph/new-host-20260919/selectivity-20261028
test -f state.txt
grep -q '^exit=1 ' state.txt
if kill -0 "$(cat queue.pid)" 2>/dev/null; then echo 'Queue still active'; exit 1; fi
exec 9>/mnt/ssd/ymy/CubeGraph/new-host-20260919/logs/runner.lock
flock -n 9
sha256sum -c source-b733dfd.tar.sha256
test ! -e code-v2
test ! -e attempts
mkdir code-v2 attempts
tar -xf source-b733dfd.tar -C code-v2
mv logs/query_acorn_sift_uniform_2d_rep1.log attempts/
mv results/acorn_sift_uniform_2d_rep1 attempts/
cp queue.log queue.pid state.txt attempts/
cp provenance/environment.txt provenance/author-source-verification.txt attempts/
python3 - <<'PYCODE'
import json
from pathlib import Path
Path('provenance/execution-revisions.json').write_text(json.dumps(dict(initial='6b3972e',instrumentation_v2='b733dfd',reason='Preserve and classify ACORN predicate-invalid cells, without changing search/indexes',initial_source='code',instrumentation_source='code-v2',replaced_binaries=['bin/acorn-v2','bin/radius_acorn-v2'],unchanged_completed=['cg_sift_uniform_2d','post_sift_uniform_2d','khi_sift_uniform_2d','irange_sift_uniform_2d']),indent=2)+'\n')
PYCODE
sha256sum /mnt/ssd/ymy/CubeGraph/external/acorn-source/faiss/impl/ACORN.cpp /mnt/ssd/ymy/CubeGraph/external/acorn-source/build/faiss/libfaiss.a >provenance/acorn-library.sha256
mv state.txt attempts/last-state.txt
printf 'RESUME_V2 %s\n' "$(date -Is)" >>queue.log
flock -u 9
nohup bash code-v2/experiments/selectivity/run.sh "$PWD" >>queue.log 2>&1 </dev/null 9>&- &
pid=$!
printf '%s\n' "$pid" >queue.pid
sleep 1
kill -0 "$pid"
cat queue.pid
tail -n 5 queue.log
