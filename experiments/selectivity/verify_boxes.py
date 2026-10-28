#!/usr/bin/env python3
"""Independent exhaustive float64 GT and stored-predicate audit for 0.5% boxes."""
import argparse
import hashlib
import json
import multiprocessing as mp
import struct
from pathlib import Path
import numpy as np

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def fvec(path):
    d = int(np.fromfile(path, dtype='<i4', count=1)[0])
    x = np.memmap(path, dtype='<f4', mode='r').reshape(-1, d + 1)
    assert np.all(x[:, 0].view('<i4') == d)
    return x[:, 1:]

def init(root, dataset, metadata):
    global X, Q, M
    p = Path(root) / dataset
    X, Q = fvec(p / f'{dataset}_base.fvecs'), fvec(p / f'{dataset}_query.fvecs')
    path = p / f'{dataset}_metadata_{metadata}.bin'
    n, d = np.fromfile(path, dtype='<u8', count=2)
    M = np.memmap(path, dtype='<f4', mode='r', offset=16, shape=(int(n), int(d)))
    assert len(X) == n and X.shape[1] == Q.shape[1]

def query(task):
    i, box, gt = task
    ids = np.flatnonzero(np.all((M >= box[0]) & (M <= box[1]), axis=1))
    assert len(ids) >= 20
    distances = []
    for start in range(0, len(ids), 8192):
        delta = X[ids[start:start+8192]].astype(np.float64) - Q[i].astype(np.float64)
        distances.append(np.einsum('ij,ij->i', delta, delta))
    dist = np.concatenate(distances)
    expected = ids[np.lexsort((ids, dist))[:20]]
    assert np.array_equal(gt, expected), (i, 'exact GT mismatch', gt.tolist(), expected.tolist())
    return len(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root', type=Path)
    ap.add_argument('dataset')
    ap.add_argument('metadata')
    ap.add_argument('workload', type=Path)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    raw = a.workload.read_bytes()
    nq, d, k = struct.unpack_from('<QQQ', raw)
    assert (nq, k) == (1000, 20) and d in (2, 3, 4)
    assert len(raw) == 24 + nq * (d * 8 + k * 8)
    boxes = np.frombuffer(raw, '<f4', nq * d * 2, 24).reshape(nq, 2, d)
    gt = np.frombuffer(raw, '<i8', nq * k, 24 + nq * d * 8).reshape(nq, k)
    assert np.isfinite(boxes).all() and np.all(boxes[:, 0] <= boxes[:, 1])
    with mp.Pool(a.workers, initializer=init, initargs=(a.data_root, a.dataset, a.metadata)) as pool:
        counts = list(pool.imap(query, [(i, boxes[i], gt[i]) for i in range(nq)], chunksize=4))
    n = int(np.fromfile(a.data_root / a.dataset / f'{a.dataset}_metadata_{a.metadata}.bin', '<u8', count=1)[0])
    # Keep the original count-calibrated protocol, reporting tied-boundary deviations.
    # Fail grossly incorrect selectivity rather than silently labeling it 0.5%.
    assert abs(np.mean(counts) / n - .005) <= .00005, (min(counts), max(counts), n)
    receipt = dict(status='PASS', workload_sha256=sha(a.workload), queries=nq,
                   exact_gt_queries=nq, target_fraction=.005, objects=n,
                   min_count=min(counts), max_count=max(counts), counts=counts,
                   mean_fraction=float(np.mean(counts) / n),
                   distance='exhaustive float64 squared L2 with object-ID tie breaking')
    Path(str(a.workload) + '.audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('GT_PASS', a.workload.name, min(counts), max(counts), receipt['mean_fraction'], flush=True)

if __name__ == '__main__':
    main()
