#!/usr/bin/env python3
"""Verify archived CSV/audit pairs and reproduce the common-recall mode table.

Optional --ids-root points at the original formal/results archive and also
checks every returned-ID file against its independent audit receipt.
"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'sources.json').read_text())
    groups = {}
    verified_ids = 0
    expected = {(r, e, p) for r in (100, 200, 500, 1000)
                for e in (20, 32, 64, 128, 256, 512, 1024, 2048)
                for p in (1, 2, 3)}
    for item in manifest['measurements']:
        path = root / item['file']
        assert sha(path) == item['sha256'], path
        audit = json.loads(path.with_suffix('.csv.audit.json').read_text())
        assert audit['status'] == 'PASS' and audit['csv_sha256'] == sha(path)
        receipts = {(r['ratio'], r['ef'], r['repeat']): r for r in audit['rows']}
        assert set(receipts) == expected
        cells, seen = {}, set()
        for row in csv.DictReader(path.open()):
            ratio, ef, rep = round(float(row['target_ratio']) * 10000), int(row['ef_search']), int(row['repeat'])
            key = ratio, ef, rep
            assert key not in seen
            seen.add(key)
            assert row['dataset'] == item['dataset']
            assert row['metadata'] == 'uniform_2d' and int(row['layers']) == 6
            assert int(row['queries']) == 1000 and int(row['k']) == 20
            assert 0 <= float(row['recall']) <= 1 and float(row['qps']) > 0
            assert abs(float(row['recall']) - receipts[key]['recall']) < 1e-9
            if args.ids_root:
                ids = args.ids_root / (item['archive_path'] + f'.r{ratio}.ef{ef}.rep{rep}.ids')
                assert ids.stat().st_size == 160000 and sha(ids) == receipts[key]['ids_sha256'], ids
                verified_ids += 1
            cells.setdefault((ratio, ef), []).append(row)
        assert seen == expected
        groups[item['dataset'], item['mode']] = (cells, receipts)
    output = []
    for ds in ('sift', 'msmarc10m'):
        pred, pr = groups[ds, 'pred']
        fly, fr = groups[ds, 'fly']
        for key in expected:
            assert pr[key]['workload_sha256'] == fr[key]['workload_sha256']
        for ratio in (100, 200, 500, 1000):
            result = {'dataset': ds, 'selectivity': ratio / 10000}
            for mode, cells in [('pred', pred), ('fly', fly)]:
                candidates = [(ef, rows) for (r, ef), rows in sorted(cells.items())
                              if r == ratio and min(float(v['recall']) for v in rows) >= .99]
                assert candidates, (ds, ratio, mode)
                ef, rows = candidates[0]
                assert len(rows) == 3
                result[mode + '_width'] = ef
                result[mode + '_qps'] = statistics.median(float(v['qps']) for v in rows)
                result[mode + '_min_recall'] = min(float(v['recall']) for v in rows)
                result[mode + '_index_bytes'] = int(rows[0]['index_bytes'])
            assert result['pred_index_bytes'] == result['fly_index_bytes']
            result['pred_over_fly'] = result['pred_qps'] / result['fly_qps']
            output.append(result)
    dest = args.output or root / 'operating-points.csv'
    with dest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({'status': 'PASS', 'csv_rows': 384, 'operating_points': 8,
                      'id_files_reverified': verified_ids, 'output': str(dest)}))


if __name__ == '__main__':
    main()
