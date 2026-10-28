#!/usr/bin/env python3
"""Fail-closed collection audit for all 78 CSVs and 1,422 measurement cells."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from verify_boxes import sha

CONFIGS = [('sift', f'uniform_{d}d') for d in (2,3,4)] + [('yfcc',f'real_{d}d') for d in (2,3)] + [('wikitext1m',f'real_{d}d') for d in (2,3)] + [('msmarc10m',f'uniform_{d}d') for d in (2,3,4)]
WIDTHS=(20,32,64,128,256,512,1024,2048)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--receipts-only',action='store_true');ap.add_argument('--output-dir',type=Path);a=ap.parse_args()
    root=a.root;output=a.output_dir or root;output.mkdir(parents=True,exist_ok=True)
    assert (root/'state.txt').read_text().startswith('exit=0 ')
    files={};measurements=[];summary=[];raw_files=0;csv_files=0;gt_queries=0
    def freeze(p):
        files[str(p.relative_to(root))]=sha(p)
    def ids(p, expected):
        nonlocal raw_files
        if not a.receipts_only:
            assert sha(p)==expected,p
            raw_files+=1
    def points(dataset,metadata,method,rows,widths=WIDTHS,predicate='box'):
        expected={(e,r) for e in widths for r in (1,2,3)}
        assert len(rows)==len(expected) and {(v['ef'],v['repeat']) for v in rows}==expected
        qualifying=[];valid_floors=[]
        for row in rows:
            row.setdefault('valid',True);row.setdefault('predicate_violations',0)
            assert bool(row['valid'])==(row['predicate_violations']==0)
        for ef in widths:
            group=[v for v in rows if v['ef']==ef]
            assert all(v['qps']>0 and 0<=v['recall']<=1 for v in group)
            low=min(v['recall'] for v in group)
            if all(v['valid'] for v in group):valid_floors.append(low)
            if all(v['valid'] for v in group) and low>=.99:qualifying.append((ef,statistics.median(v['qps'] for v in group),low))
        point=dict(dataset=dataset,metadata=metadata,predicate=predicate,method=method,ratio=.005,ef='',qps='',min_recall='',highest_min_recall=max(valid_floors) if valid_floors else '',highest_observed_min_recall=max(min(v['recall'] for v in rows if v['ef']==ef) for ef in widths),valid_widths=len(valid_floors),invalid_cells=sum(not v['valid'] for v in rows))
        if qualifying:
            ef,qps,recall=min(qualifying);point.update(ef=ef,qps=qps,min_recall=recall)
        summary.append(point)
        for row in rows:measurements.append(dict(dataset=dataset,metadata=metadata,predicate=predicate,method=method,ratio=.005,**row))
    for ds,meta in CONFIGS:
        tag=f'{ds}_{meta}';wl=root/'workloads'/tag/f'{tag}_r0050.bin';auditpath=Path(str(wl)+'.audit.json')
        audit=json.loads(auditpath.read_text());assert audit['status']=='PASS' and audit['workload_sha256']==sha(wl) and audit['exact_gt_queries']==1000
        assert audit['target_fraction']==.005 and len(audit['counts'])==1000
        freeze(wl);freeze(auditpath);freeze(wl.parent/'workload_stats.csv');gt_queries+=1000
        cv=root/'converted'/tag/'complete.json';conversion=json.loads(cv.read_text());assert conversion['status']=='PASS' and conversion['workload_sha256']==sha(wl);freeze(cv)
        for name,h in conversion['generated_sha256'].items():
            p=cv.parent/name;assert sha(p)==h;freeze(p)
        for method in ['cg','post']+(['fly'] if ds in ('sift','msmarc10m') and meta=='uniform_2d' else []):
            p=root/'results'/f'{method}_{tag}.csv';apath=Path(str(p)+'.audit.json');audit=json.loads(apath.read_text());assert audit['status']=='PASS' and audit['csv_sha256']==sha(p)
            source=list(csv.DictReader(p.open()));assert len(audit['rows'])==len(source)==24
            checks={(v['ef'],v['repeat']):v for v in audit['rows']};assert len(checks)==24
            rows=[]
            for v in source:
                ef=int(v['budget'] if method=='post' else v['ef_search']);rep=int(v['repeat']);check=checks[ef,rep]
                assert v['dataset']==ds and v['metadata']==meta and float(v['target_ratio'])==.005 and int(v['queries'])==1000 and int(v['k'])==20
                assert check['ratio']==50 and check['workload_sha256']==sha(wl) and abs(check['recall']-float(v['recall']))<1e-8
                ids(Path(str(p)+f'.r50.ef{ef}.rep{rep}.ids'),check['ids_sha256'])
                rows.append(dict(ef=ef,repeat=rep,recall=float(v['recall']),qps=float(v['qps'])))
            freeze(p);freeze(apath);csv_files+=1;points(ds,meta,method,rows)
        for method in ['khi']+(['irange'] if meta.endswith('2d') else []):
            p=root/'results'/f'{method}_{tag}.csv';apath=Path(str(p)+'.audit.json');audit=json.loads(apath.read_text())
            assert audit['pass'] and audit['parameter_rows']==24 and audit['query_outputs_checked']==24000 and audit['csv_sha256']==sha(p)
            ids(Path(str(p)+'.ids.tsv'),audit['ids_sha256'])
            source=list(csv.DictReader(p.open()));rows=[]
            for v in source:
                assert v['suite']=='fixedcard' and v['tag']=='r0050' and int(v['queries'])==1000 and int(v['k'])==20 and v['validated']=='1'
                rows.append(dict(ef=int(v['ef']),repeat=int(v['repeat']),recall=float(v['recall']),qps=float(v['qps'])))
            freeze(p);freeze(apath);csv_files+=1;points(ds,meta,method,rows)
        rows=[]
        for rep in (1,2,3):
            p=root/'results'/f'acorn_{tag}_rep{rep}'/f'{ds}_M32_g12_mb64_{meta}.csv';apath=Path(str(p)+'.audit.json');audit=json.loads(apath.read_text());assert audit['status'] in ('PASS','CHECKED_WITH_INVALID_CELLS') and audit['csv_sha256']==sha(p)
            source=list(csv.DictReader(p.open()));assert len(audit['rows'])==len(source)==9
            assert audit['invalid_cells']==sum(not c['valid'] for c in audit['rows'])
            assert (audit['status']=='PASS')==(audit['invalid_cells']==0)
            checks={v['ef']:v for v in audit['rows']};assert len(checks)==9
            for v in source:
                ef=int(v['ef_search']);check=checks[ef]
                assert v['dataset']==ds and v['metadata']==meta and float(v['target_ratio'])==.005 and int(v['queries'])==1000 and int(v['k'])==20 and int(v['gamma'])==12
                assert check['ratio']==50 and check['workload_sha256']==sha(wl) and abs(check['recall']-float(v['recall']))<1e-8
                assert int(v['valid'])==int(check['valid']) and int(v['predicate_violations'])==check['predicate_violations']
                assert check['result_status']==('PASS' if check['valid'] else 'FAIL_PREDICATE')
                ids(Path(str(p)+f'.r50.ef{ef}.ids'),check['ids_sha256']);rows.append(dict(ef=ef,repeat=rep,recall=float(v['recall']),qps=float(v['qps']),valid=check['valid'],predicate_violations=check['predicate_violations']))
            freeze(p);freeze(apath);csv_files+=1
        points(ds,meta,'acorn',rows,WIDTHS+(4096,))
    apath=root/'radius/results/audit.json';audit=json.loads(apath.read_text());assert audit['status'] in ('PASS','CHECKED_WITH_INVALID_CELLS') and len(audit['workloads'])==4 and len(audit['results'])==12;freeze(apath)
    for name,info in audit['workloads'].items():
        wl=root/'radius/workloads'/name;assert sha(wl)==info['sha256'] and info['queries']==info['full_bruteforce_queries']==1000;freeze(wl);freeze(wl.with_suffix('.queries.csv'));gt_queries+=1000
        for method in ('cg','post','acorn'):
            p=root/'radius/results'/f'{method}_{wl.stem}.csv';v=audit['results'][p.name];assert v['sha256']==sha(p) and v['rows']==24 and len(v['ids_sha256'])==24
            for filename,h in v['ids_sha256'].items():ids(p.parent/filename,h)
            source=list(csv.DictReader(p.open()));rows=[]
            checks={(c['ef'],c['repeat']):c for c in v['cells']};assert len(checks)==24
            assert v['invalid_cells']==sum(not c['valid'] for c in v['cells'])
            for row in source:
                assert float(row['target_ratio'])==.005 and row['method']==method and int(row['queries'])==1000 and int(row['k'])==20
                ef,rep=int(row['search_width']),int(row['repeat']);check=checks[ef,rep]
                assert check['result_status']==('PASS' if check['valid'] else 'FAIL_PREDICATE')
                if method!='acorn':assert check['valid']
                assert int(row.get('valid',1))==int(check['valid']) and int(row.get('predicate_violations',0))==check['predicate_violations']
                assert abs(check['recall']-float(row['recall']))<1e-8
                rows.append(dict(ef=ef,repeat=rep,recall=float(row['recall']),qps=float(row['qps']),valid=check['valid'],predicate_violations=check['predicate_violations']))
            freeze(p);csv_files+=1;points('sift',f'uniform_{source[0]["dimension"]}d',method,rows,predicate=source[0]['predicate'])
    assert audit['invalid_cells']==sum(v['invalid_cells'] for v in audit['results'].values())
    assert (audit['status']=='PASS')==(audit['invalid_cells']==0)
    assert (csv_files,len(measurements),len(summary),gt_queries)==(78,1422,58,14000)
    assert raw_files==(0 if a.receipts_only else 1100)
    for p in (root/'provenance').glob('*'):
        if p.is_file():freeze(p)
    for p in (root/'state.txt',root/'queue.log',root/'source-commit.txt'):freeze(p)
    for name,rows in [('summary.csv',summary),('measurements.csv',measurements)]:
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    receipt=dict(status='PASS',invalid_measurement_cells=sum(not v['valid'] for v in measurements),csv_files=csv_files,measurement_rows=len(measurements),operating_points=len(summary),ground_truth_queries=gt_queries,raw_id_files_verified=raw_files,receipts_only=a.receipts_only,source_sha256=files,summary_sha256=sha(output/'summary.csv'),measurements_sha256=sha(output/'measurements.csv'))
    (output/'collection-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
