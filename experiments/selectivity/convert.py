#!/usr/bin/env python3
"""Add exact 0.5% constraints to a fresh directory referencing frozen native data."""
import argparse
import json
import struct
from pathlib import Path
import numpy as np
from verify_boxes import fvec, sha

def columns(path, names, data):
    with path.open('wb') as f:
        f.write(struct.pack('<i', len(names)))
        for name in names:
            b = name.encode(); f.write(struct.pack('<i', len(b))); f.write(b)
        f.write(struct.pack('<i', len(data))); data.T.astype('<f4').tofile(f)

def read_columns(path):
    with path.open('rb') as f:
        d = struct.unpack('<i', f.read(4))[0]
        for _ in range(d):
            n = struct.unpack('<i', f.read(4))[0]; f.read(n)
        n = struct.unpack('<i', f.read(4))[0]
        return np.fromfile(f, '<f4').reshape(d, n).T

def main():
    ap = argparse.ArgumentParser()
    for name in ('data_root', 'dataset', 'metadata', 'workload', 'original', 'output'):
        ap.add_argument(name, type=str if name in ('dataset', 'metadata') else Path)
    a = ap.parse_args(); ds, md = a.dataset, a.metadata
    receipt = json.loads(Path(str(a.workload) + '.audit.json').read_text())
    assert receipt['status'] == 'PASS' and receipt['workload_sha256'] == sha(a.workload)
    base = fvec(a.data_root / ds / f'{ds}_base.fvecs')
    queries = fvec(a.data_root / ds / f'{ds}_query.fvecs')[:1000]
    raw = a.workload.read_bytes(); nq, d, k = struct.unpack_from('<QQQ', raw)
    assert (nq, k) == (1000, 20)
    boxes = np.frombuffer(raw, '<f4', nq*d*2, 24).reshape(nq, 2, d)
    gt = np.frombuffer(raw, '<i8', nq*k, 24+nq*d*8).reshape(nq, k)
    meta = np.memmap(a.data_root / ds / f'{ds}_metadata_{md}.bin', '<f4', mode='r', offset=16).reshape(-1, d)
    assert np.array_equal(read_columns(a.original/'meta.bin'), meta)
    vectors = np.memmap(a.original/'vectors.bin', '<f4', mode='r', offset=8).reshape(base.shape)
    for start in range(0, len(base), 10000):
        assert np.array_equal(vectors[start:start+10000], base[start:start+10000])
    q = np.memmap(a.original/'queries.bin', '<f4', mode='r', offset=8).reshape(queries.shape)
    assert np.array_equal(q, queries)
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ['vectors.bin', 'queries.bin', 'meta.bin'] + (['attr0.bin', 'attr1.bin'] if d == 2 else []):
        (a.output/name).symlink_to((a.original/name).resolve())
    cols = np.stack([boxes[:, 0], boxes[:, 1]], axis=2).reshape(nq, 2*d)
    columns(a.output/'fixedcard_r0050_constraints.bin',
            [s for j in range(d) for s in (f'a{j}_low', f'a{j}_high')], cols)
    with (a.output/'fixedcard_r0050_gt.bin').open('wb') as f:
        f.write(struct.pack('<ii', nq, k))
        for i, ids in enumerate(gt):
            assert np.all((ids >= 0) & (ids < len(meta)))
            assert np.all((meta[ids] >= boxes[i, 0]) & (meta[ids] <= boxes[i, 1]))
            ids.astype('<i4').tofile(f)
            np.sum((base[ids]-queries[i])**2, axis=1).astype('<f4').tofile(f)
    if d == 2:
        bounds = np.empty((nq, d, 2), '<i4')
        for j in range(d):
            unique = np.unique(meta[:, j])
            ranks = np.searchsorted(unique, meta[:, j]).astype('<i4')
            assert np.array_equal(ranks, np.fromfile(a.original/f'attr{j}.bin', '<i4'))
            bounds[:, j, 0] = np.searchsorted(unique, boxes[:, 0, j], side='left')
            bounds[:, j, 1] = np.searchsorted(unique, boxes[:, 1, j], side='right')-1
            for i in range(nq):
                lo, hi = bounds[i, j]
                assert (lo == len(unique) or unique[lo] >= boxes[i, 0, j]) and (lo == 0 or unique[lo-1] < boxes[i, 0, j])
                assert (hi < 0 or unique[hi] <= boxes[i, 1, j]) and (hi+1 == len(unique) or unique[hi+1] > boxes[i, 1, j])
        bounds.tofile(a.output/'fixedcard_r0050_ranges.bin')
    result = dict(status='PASS', dataset=ds, metadata=md, workload_sha256=sha(a.workload),
                  vectors_and_query_order_verified=True, metadata_verified=True, integer_bounds_exact=d==2,
                  generated_sha256={p.name:sha(p) for p in a.output.glob('fixedcard*')})
    (a.output/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CONVERT_PASS', ds, md, flush=True)

if __name__ == '__main__':
    main()
