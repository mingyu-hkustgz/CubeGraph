#!/usr/bin/env python3
"""Independent NumPy exact oracle; does not import benchmark predicates or GT code."""
import argparse,csv,json,multiprocessing as mp,struct,hashlib
from pathlib import Path
import numpy as np
from audit_validity import check_ids

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read_workload(p):
 with open(p,'rb') as f:
  assert f.read(8)==b'RCWL0001'
  nq,d,kind,k=struct.unpack('<4Q',f.read(32));ratio=struct.unpack('<d',f.read(8))[0]
  params=np.fromfile(f,dtype='<f4',count=nq*13).reshape(nq,13)
  counts=np.fromfile(f,dtype='<u8',count=nq);gt=np.fromfile(f,dtype='<i8',count=nq*k).reshape(nq,k)
  assert not f.read(1)
 return nq,d,kind,k,ratio,params,counts,gt

def predicate(m,p,d,kind):
 # Explicit float32 reduction matches the published on-disk predicate semantics.
 dist=np.zeros(len(m),dtype=np.float32)
 for j in range(d):
  delta=m[:,j]-p[j];dist+=delta*delta
 ball=dist<=np.float32(p[12]*p[12])
 return ball if kind==0 else (~ball)&np.all((m>=p[4:4+d])&(m<=p[8:8+d]),axis=1)

def init(data,d):
 global X,Q,M
 def vec(p):
  dim=int(np.fromfile(p,dtype='<i4',count=1)[0]);return np.memmap(p,dtype='<f4',mode='r').reshape(-1,dim+1)[:,1:]
 X=vec(data/'sift_base.fvecs');Q=vec(data/'sift_query.fvecs')
 p=data/f'sift_metadata_uniform_{d}d.bin';n,md=np.fromfile(p,dtype='<u8',count=2);assert md==d
 M=np.memmap(p,dtype='<f4',offset=16,mode='r',shape=(int(n),d))

def check_query(args):
 qi,d,kind,p,count,gt=args
 ids=np.flatnonzero(predicate(M,p,d,kind));assert len(ids)==count,(qi,'count',len(ids),int(count))
 if kind:
  inbox=np.all((M>=p[4:4+d])&(M<=p[8:8+d]),axis=1)
  assert len(ids)<int(inbox.sum()) and int(inbox.sum())>0,(qi,'degenerate exclusion')
  assert np.all(p[:d]>=p[4:4+d]) and np.all(p[:d]<=p[8:8+d])
 delta=X[ids].astype(np.float64)-Q[qi].astype(np.float64)
 dist=np.einsum('ij,ij->i',delta,delta)
 order=np.lexsort((ids,dist))[:20];expected=np.full(20,-1,dtype=np.int64);expected[:len(order)]=ids[order]
 assert np.array_equal(gt,expected),(qi,'GT mismatch',gt.tolist(),expected.tolist())
 return int(len(ids))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('data',type=Path);ap.add_argument('workloads',type=Path);ap.add_argument('results',type=Path);ap.add_argument('--workers',type=int,default=16);ap.add_argument('--smoke',action='store_true');ap.add_argument('--gt-only',action='store_true');a=ap.parse_args();report={'status':'PASS','workloads':{},'results':{},'problems':[]}
 try:
  for path in sorted(a.workloads.glob('*.bin')):
   if '.ids.' in path.name:continue
   nq,d,kind,k,ratio,params,counts,gt=read_workload(path)
   if not a.smoke:assert nq==1000 and ratio==.005
   tasks=[(i,d,kind,params[i],counts[i],gt[i]) for i in range(nq)]
   with mp.Pool(a.workers,initializer=init,initargs=(a.data,d)) as pool:
    checked=list(pool.imap(check_query,tasks,chunksize=4))
   init(a.data,d)
   report['workloads'][path.name]={'sha256':sha(path),'queries':nq,'min_eligible':min(checked),'max_eligible':max(checked),'measured_selectivity':sum(checked)/(nq*len(M)),'full_bruteforce_queries':nq}
   print('GT_PASS',path.name,flush=True)
   if a.gt_only:continue
   for method in ('cg','post','acorn'):
    csvpath=a.results/f'{method}_{path.stem}.csv';rows=list(csv.DictReader(csvpath.open()));expected={(ef,rep) for ef in ([20,2048] if a.smoke else [20,32,64,128,256,512,1024,2048]) for rep in ([1] if a.smoke else [1,2,3])};seen=set();artifacts={};checks=[]
    for r in rows:
     ef,rep=int(r['search_width']),int(r['repeat']);assert (ef,rep) not in seen;seen.add((ef,rep))
     assert int(r['queries'])==nq and int(r['k'])==k and int(r['dimension'])==d and r['method']==method and r['predicate']==('compose' if kind else 'radius') and float(r['target_ratio'])==ratio
     assert abs(float(r['measured_selectivity'])-counts.sum()/(nq*len(M)))<1e-10
     idpath=Path(str(csvpath)+f'.ef{ef}.rep{rep}.ids.bin');ids=np.fromfile(idpath,dtype='<i8').reshape(nq,k);hits=0;denom=int((gt>=0).sum())
     check=check_ids(ids,gt,len(M),lambda qi,got: predicate(M[got],params[qi],d,kind))
     if method!='acorn':assert check['valid'],(idpath,check)
     if 'valid' in r:
      assert int(r['valid'])==int(check['valid']) and int(r['predicate_violations'])==check['predicate_violations']
     else:assert check['valid'], 'unclassified invalid measurement'
     assert abs(check['recall']-float(r['recall']))<1e-10
     checks.append(dict(ef=ef,repeat=rep,**check))
     qps=float(r['qps']);seconds=float(r['search_seconds']);assert np.isfinite(qps) and qps>0 and abs(qps-nq/seconds)/qps<1e-9
     assert 0<=float(r['p50_us'])<=float(r['p95_us']) and float(r['prep_us_mean'])>=0
     artifacts[idpath.name]=sha(idpath)
    assert seen==expected,(csvpath,'missing cells',expected-seen)
    report['results'][csvpath.name]={'sha256':sha(csvpath),'rows':len(rows),'ids_sha256':artifacts,'cells':checks,'invalid_cells':sum(not c['valid'] for c in checks)};print('RESULT_PASS',csvpath.name,flush=True)
  if not a.smoke:assert len(report['workloads'])==4
  report['invalid_cells']=sum(v['invalid_cells'] for v in report['results'].values())
  if report['invalid_cells']:report['status']='CHECKED_WITH_INVALID_CELLS'
 except Exception as e:
  report['status']='FAIL';report['problems'].append(repr(e));raise
 finally:
  a.results.mkdir(parents=True,exist_ok=True);(a.results/('gt-audit.json' if a.gt_only else 'audit.json')).write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
