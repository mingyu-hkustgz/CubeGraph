#!/usr/bin/env python3
"""Freeze count-calibrated 2D boxes at aspect ratios 1, 4, 16, and 32.

The primary protocol targets 1% eligible objects, avoiding the geometric
infeasibility of a contained 10%-area rectangle with aspect ratio 32.
All shapes share query vectors and sampled anchors. Anchors infeasible for
any shape are rejected jointly, before any search measurement is observed.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

ASPECTS = (1, 4, 16, 32)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fvecs(path):
    with path.open('rb') as f:
        d = struct.unpack('<i', f.read(4))[0]
    assert d > 0 and path.stat().st_size % (4 * (d + 1)) == 0
    mm = np.memmap(path, dtype='<f4', mode='r').reshape(-1, d + 1)
    assert np.all(mm[:, 0].view('<i4') == d)
    return mm[:, 1:]


def interval_threshold(points, center):
    """Smallest width of a nested, domain-contained interval including x.

    The interval is centered at center until touching an edge, then translated
    to remain within [0,1]. This expression avoids repeated count bisections.
    """
    return np.where(points >= center,
                    np.minimum(2 * (points - center), points),
                    np.minimum(2 * (center - points), 1 - points))


def bounds(center, width, alpha, axis):
    sides = np.full(2, width / alpha, dtype=np.float64)
    sides[axis] = width
    lower = np.maximum(0, np.minimum(center - sides / 2, 1 - sides))
    return np.array([lower, lower + sides], dtype='<f4')


def members(attrs, box):
    return np.flatnonzero(np.all((attrs >= box[0]) & (attrs <= box[1]), axis=1))


def calibrated_box(attrs, thresholds, center, target, alpha, axis):
    scaled = thresholds.copy()
    scaled[:, 1 - axis] *= alpha
    need = scaled.max(axis=1)
    width = float(np.partition(need, target - 1)[target - 1])
    if width > 1 or width <= 0:
        return None
    box = bounds(center, width, alpha, axis)
    ids = members(attrs, box)
    # float32 bounds and tied real attributes can change a boundary count.
    # Accept only <=0.1% cardinality error; reject this anchor for ALL shapes.
    if abs(len(ids) - target) > max(2, int(target * .001)):
        return None
    return box, ids


def exact_gt(vectors, query, ids, k):
    # Exhaustive float64 squared L2, stable object-ID tie breaking.
    delta = np.asarray(vectors[ids], dtype=np.float64) - query.astype(np.float64)
    dist = np.einsum('ij,ij->i', delta, delta)
    return ids[np.lexsort((ids, dist))[:k]].astype('<i8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_root', type=Path)
    ap.add_argument('dataset', choices=('sift', 'yfcc'))
    ap.add_argument('output', type=Path)
    ap.add_argument('--queries', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261027)
    args = ap.parse_args()
    assert args.queries > 0
    meta = 'uniform_2d' if args.dataset == 'sift' else 'real_2d'
    root = args.data_root / args.dataset
    paths = {kind: root / f'{args.dataset}_{name}' for kind, name in
             [('base', 'base.fvecs'), ('query', 'query.fvecs'),
              ('metadata', f'metadata_{meta}.bin')]}
    with paths['metadata'].open('rb') as f:
        n, d = struct.unpack('<QQ', f.read(16))
    assert d == 2 and paths['metadata'].stat().st_size == 16 + n * d * 4
    raw_attrs = np.memmap(paths['metadata'], dtype='<f4', mode='r', offset=16).reshape(n, d)
    assert np.isfinite(raw_attrs).all()
    domain_min = raw_attrs.min(axis=0).astype(float)
    domain_max = raw_attrs.max(axis=0).astype(float)
    scale = domain_max - domain_min
    assert np.all(scale > 0)
    attrs = (raw_attrs.astype(float) - domain_min) / scale
    vectors, queries = fvecs(paths['base']), fvecs(paths['query'])
    assert len(vectors) == n and len(queries) >= args.queries
    assert vectors.shape[1] == queries.shape[1]
    target, k = round(n * .01), 20
    assert target >= k
    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    boxes = {a: [] for a in ASPECTS}
    gt = {a: [] for a in ASPECTS}
    details, attempts = [], 0
    while len(details) < args.queries:
        attempts += 1
        if attempts > 100 * args.queries:
            raise RuntimeError('Too few jointly feasible anchors; no workloads published')
        qi = len(details)
        center = np.asarray(attrs[rng.integers(n)], dtype=np.float64)
        axis = qi % 2
        thresholds = np.column_stack([interval_threshold(attrs[:, j].astype(np.float64), center[j])
                                      for j in range(2)])
        candidates = {a: calibrated_box(attrs, thresholds, center, target, a, axis) for a in ASPECTS}
        if any(v is None for v in candidates.values()):
            continue
        # Store predicates in the original metadata coordinates, which the
        # index and independent returned-ID auditor consume.
        for a, (box, _) in candidates.items():
            raw_box = (box.astype(float) * scale + domain_min).astype('<f4')
            candidates[a] = raw_box, members(raw_attrs, raw_box)
        if any(abs(len(v[1]) - target) > max(2, int(target * .001)) for v in candidates.values()):
            continue
        record = {'query': qi, 'anchor': center.tolist(), 'long_axis': axis, 'counts': {}}
        for a, (box, ids) in candidates.items():
            boxes[a].append(box)
            gt[a].append(exact_gt(vectors, queries[qi], ids, k))
            record['counts'][str(a)] = len(ids)
        details.append(record)
        if len(details) % 50 == 0:
            print(f'generated {len(details)}/{args.queries}; attempts={attempts}', flush=True)
    manifest = {'protocol': 'count-calibrated-contained-2d-boxes-v1', 'dataset': args.dataset,
                'metadata': meta, 'seed': args.seed, 'queries': args.queries, 'k': k,
                'target_fraction': .01, 'target_count': target, 'aspects': ASPECTS,
                'domain_min': domain_min.tolist(), 'domain_max': domain_max.tolist(),
                'aspect_coordinates': 'per-attribute min/max normalized domain',
                'anchor_sampling': 'uniform over object IDs; jointly reject infeasible anchors',
                'attempted_anchors': attempts, 'accepted_anchors': len(details),
                'distance': 'exhaustive float64 squared L2; object-ID tie breaking',
                'inputs': {name: {'file': p.name, 'sha256': sha(p)} for name, p in paths.items()},
                'workloads': [], 'query_details': details}
    for a in ASPECTS:
        folder = args.output / f'a{a}'
        folder.mkdir()
        p = folder / f'{args.dataset}_{meta}_r0100.bin'
        with p.open('wb') as f:
            f.write(struct.pack('<QQQ', args.queries, 2, k))
            f.write(np.asarray(boxes[a], dtype='<f4').tobytes())
            f.write(np.asarray(gt[a], dtype='<i8').tobytes())
        manifest['workloads'].append({'aspect': a, 'file': str(p.relative_to(args.output)), 'sha256': sha(p)})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('WORKLOAD_GENERATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
