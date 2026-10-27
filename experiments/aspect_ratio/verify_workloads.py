#!/usr/bin/env python3
"""Independently scan all eligible vectors to check frozen aspect workloads."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from generate import fvecs, sha

ap = argparse.ArgumentParser()
ap.add_argument('data_root', type=Path)
ap.add_argument('workloads', type=Path)
args = ap.parse_args()
m = json.loads((args.workloads / 'manifest.json').read_text())
root = args.data_root / m['dataset']
for item in m['inputs'].values():
    assert sha(root / item['file']) == item['sha256']
meta = root / m['inputs']['metadata']['file']
with meta.open('rb') as f:
    n, d = struct.unpack('<QQ', f.read(16))
attrs = np.memmap(meta, dtype='<f4', mode='r', offset=16).reshape(n, d)
base = fvecs(root / m['inputs']['base']['file'])
queries = fvecs(root / m['inputs']['query']['file'])
for entry in m['workloads']:
    path = args.workloads / entry['file']
    assert sha(path) == entry['sha256']
    raw = path.read_bytes()
    nq, wd, k = struct.unpack_from('<QQQ', raw)
    assert (nq, wd, k) == (m['queries'], 2, 20)
    assert len(raw) == 24 + nq * 16 + nq * k * 8
    boxes = np.frombuffer(raw, '<f4', count=nq * 4, offset=24).reshape(nq, 2, 2)
    gt = np.frombuffer(raw, '<i8', offset=24 + nq * 16).reshape(nq, k)
    domain_min, domain_max = np.array(m['domain_min']), np.array(m['domain_max'])
    assert np.array_equal(attrs.min(axis=0), domain_min)
    assert np.array_equal(attrs.max(axis=0), domain_max)
    sides = (boxes[:, 1].astype(float) - boxes[:, 0]) / (domain_max - domain_min)
    assert np.all(boxes >= domain_min) and np.all(boxes <= domain_max) and np.all(sides > 0)
    assert np.allclose(sides.max(axis=1) / sides.min(axis=1), entry['aspect'], rtol=1e-4)
    for qi in range(nq):
        eligible = np.flatnonzero((attrs[:, 0] >= boxes[qi, 0, 0]) &
                                 (attrs[:, 0] <= boxes[qi, 1, 0]) &
                                 (attrs[:, 1] >= boxes[qi, 0, 1]) &
                                 (attrs[:, 1] <= boxes[qi, 1, 1]))
        assert len(eligible) == m['query_details'][qi]['counts'][str(entry['aspect'])]
        assert abs(len(eligible) - m['target_count']) <= max(2, int(.001 * m['target_count']))
        # Independent accumulation, rather than the generator's einsum path.
        distance = np.zeros(len(eligible), dtype=np.float64)
        for dim in range(base.shape[1]):
            delta = base[eligible, dim].astype(float) - float(queries[qi, dim])
            distance += delta * delta
        order = np.lexsort((eligible, distance))[:k]
        assert np.array_equal(eligible[order], gt[qi]), (entry['aspect'], qi)
    print('EXACT_GT_PASS', entry['aspect'], nq, flush=True)
(args.workloads / 'verification.json').write_text(json.dumps({
    'status': 'PASS', 'workloads': len(m['workloads']), 'queries_per_workload': m['queries'],
    'manifest_sha256': sha(args.workloads / 'manifest.json')}, indent=2) + '\n')
