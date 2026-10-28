#!/usr/bin/env python3
"""Verify a collected formal aspect/mode batch and derive operating points.

The default verifies every raw returned-ID hash. --receipts-only supports the
compact public evidence copy and explicitly reports that IDs were not re-read.
"""
import argparse
import csv
import hashlib
import json
import statistics
import struct
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('--receipts-only', action='store_true')
    ap.add_argument('--output-dir', type=Path, help='Write derived files here; leave public evidence untouched')
    args = ap.parse_args()
    output = args.output_dir or args.root
    output.mkdir(parents=True, exist_ok=True)
    expected = {(e, p) for e in (20, 32, 64, 128, 256, 512, 1024, 2048) for p in (1, 2, 3)}
    records, sources, ids_checked = [], {}, 0
    counts = {}
    core = {'include/IndexCube.h': '340381c15d04037f34547288bcff318d5445ceca5bc4bec7ba1b1bdc880a3977',
            'third_party/hnsw-cube.h': '3edc62462112a07c0269bdfa6a4797681d2f7bc4daba9643008b299e5d00df07'}
    for dataset in ('sift', 'yfcc'):
        folder = args.root / dataset
        state = folder / 'state.txt'
        assert state.read_text().startswith('exit=0 '), state
        sources[str(state.relative_to(args.root))] = sha(state)
        for p in sorted((folder / 'provenance').glob('*.sha256')):
            sources[str(p.relative_to(args.root))] = sha(p)
        code_hashes = (folder / 'provenance/code.sha256').read_text().splitlines()
        for filename, digest in core.items():
            entries = [line.split()[0] for line in code_hashes if line.endswith('/' + filename)]
            assert entries == [digest], filename
        manifest_path = folder / 'workloads/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        verified = json.loads((folder / 'workloads/verification.json').read_text())
        for p in (manifest_path, folder / 'workloads/verification.json'):
            sources[str(p.relative_to(args.root))] = sha(p)
        assert verified['status'] == 'PASS' and verified['manifest_sha256'] == sha(manifest_path)
        assert manifest['queries'] == 1000 and manifest['k'] == 20
        assert manifest['aspects'] == [1, 4, 16, 32] and manifest['dataset'] == dataset
        assert manifest['accepted_anchors'] == 1000 and len(manifest['query_details']) == 1000
        assert manifest['target_fraction'] == .01
        if 'rejected_anchors' in manifest:
            assert sum(manifest['rejected_anchors'].values()) + 1000 == manifest['attempted_anchors']
            assert manifest['stored_aspect_relative_tolerance'] == 1e-4
        assert len(manifest['workloads']) == 4 and {w['aspect'] for w in manifest['workloads']} == {1, 4, 16, 32}
        assert verified['workloads'] == 4 and verified['queries_per_workload'] == 1000
        counts[dataset] = {'accepted': manifest['accepted_anchors'], 'attempted': manifest['attempted_anchors']}
        index_sizes = set()
        for workload in manifest['workloads']:
            alpha = workload['aspect']
            path = folder / 'workloads' / workload['file']
            assert sha(path) == workload['sha256']
            assert struct.unpack_from('<QQQ', path.read_bytes()) == (1000, 2, 20)
            sources[str(path.relative_to(args.root))] = sha(path)
            counts[dataset][str(alpha)] = [min(q['counts'][str(alpha)] for q in manifest['query_details']),
                                          max(q['counts'][str(alpha)] for q in manifest['query_details'])]
            for mode in ('pred', 'fly'):
                path = folder / f'results/a{alpha}_{mode}.csv'
                receipt_path = Path(str(path) + '.audit.json')
                receipt = json.loads(receipt_path.read_text())
                assert receipt['status'] == 'PASS' and receipt['csv_sha256'] == sha(path)
                sources[str(path.relative_to(args.root))] = sha(path)
                sources[str(receipt_path.relative_to(args.root))] = sha(receipt_path)
                assert Path(str(path) + '.complete').exists()
                audit = {(r['ef'], r['repeat']): r for r in receipt['rows']}
                assert len(receipt['rows']) == 24 and set(audit) == expected
                cells, keys = {}, set()
                for row in csv.DictReader(path.open()):
                    ef, rep = int(row['ef_search']), int(row['repeat'])
                    assert (ef, rep) not in keys
                    keys.add((ef, rep))
                    assert row['dataset'] == dataset and row['metadata'] == manifest['metadata']
                    assert float(row['target_ratio']) == .01 and int(row['layers']) == 6
                    index_sizes.add(int(row['index_bytes']))
                    assert int(row['queries']) == 1000 and int(row['k']) == 20
                    assert float(row['qps']) > 0 and 0 <= float(row['recall']) <= 1
                    a = audit[ef, rep]
                    assert a['ratio'] == 100 and a['workload_sha256'] == workload['sha256']
                    assert abs(float(row['recall']) - a['recall']) < 1e-9
                    if not args.receipts_only:
                        ids = Path(str(path) + f'.r100.ef{ef}.rep{rep}.ids')
                        assert ids.stat().st_size == 160000 and sha(ids) == a['ids_sha256']
                        ids_checked += 1
                    cells.setdefault(ef, []).append(row)
                assert keys == expected
                candidates = [(ef, rows) for ef, rows in sorted(cells.items())
                              if min(float(r['recall']) for r in rows) >= .99]
                point = {'dataset': dataset, 'aspect': alpha, 'mode': mode,
                         'ef': None, 'qps': None, 'min_recall': None,
                         'highest_min_recall': max(min(float(r['recall']) for r in rows) for rows in cells.values())}
                if candidates:
                    ef, rows = candidates[0]
                    point.update(ef=ef, qps=statistics.median(float(r['qps']) for r in rows),
                                 min_recall=min(float(r['recall']) for r in rows))
                records.append(point)
        assert len(index_sizes) == 1
    assert len(records) == 16 and ids_checked == (0 if args.receipts_only else 384)
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    verification = {'status': 'PASS', 'measurement_rows': 384, 'operating_points': len(records),
                    'raw_id_files_verified': ids_checked, 'receipts_only': args.receipts_only,
                    'source_sha256': sources, 'anchor_and_qualifying_counts': counts,
                    'summary_sha256': sha(output / 'summary.csv'), 'core_sha256': core}
    (output / 'collection-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({k: v for k, v in verification.items() if k not in ('source_sha256', 'anchor_and_qualifying_counts')}, indent=2))


if __name__ == '__main__':
    main()
