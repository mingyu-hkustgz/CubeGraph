import csv,struct,sys,collections,json,hashlib
from pathlib import Path
import numpy as np
src=Path(sys.argv[1]);p=Path(sys.argv[2])
def columns(p):
 with p.open('rb') as f:
  d=struct.unpack('<i',f.read(4))[0]
  for _ in range(d):n=struct.unpack('<i',f.read(4))[0];f.read(n)
  n=struct.unpack('<i',f.read(4))[0];return np.fromfile(f,'<f4').reshape(d,n).T
meta=columns(src/'meta.bin');gts={};bounds={}
for s in ['fixedcard']:
 for t in ['r0050']:
  pre=f'{s}_{t}';bounds[(s,t)]=columns(src/f'{pre}_constraints.bin')
  with (src/f'{pre}_gt.bin').open('rb') as f:
   n,k=struct.unpack('<ii',f.read(8));assert(n,k)==(1000,20);a=[]
   for _ in range(n):a.append(np.fromfile(f,'<i4',k));f.read(4*k)
  gts[(s,t)]=a
counts=collections.defaultdict(lambda:[0,0,set()])
with Path(str(p)+'.ids.tsv').open() as f:
 for r in csv.DictReader(f,delimiter='\t'):
  s,t=r['suite'],r['tag'];key=(s,t,int(r['ef']),int(r['repeat']));qi=int(r['query']);assert 0<=qi<1000
  ids=list(map(int,r['ids'].split()));assert len(ids)<=20 and len(ids)==len(set(ids));assert all(0<=i<len(meta) for i in ids)
  b=bounds[(s,t)][qi];assert np.all(meta[ids]>=b[::2]) and np.all(meta[ids]<=b[1::2])
  v=counts[key];assert qi not in v[2];v[2].add(qi);v[0]+=len(set(ids)&set(gts[(s,t)][qi]));v[1]+=len(ids)
rows=list(csv.DictReader(p.open()));assert len(rows)==len(counts)==24
for r in rows:
 v=counts[(r['suite'],r['tag'],int(r['ef']),int(r['repeat']))];assert len(v[2])==1000;assert abs(float(r['recall'])-v[0]/20000)<1e-9;assert int(r['returned'])==v[1];assert np.isfinite(float(r['qps'])) and float(r['qps'])>0 and r['validated']=='1'
assert set(counts)=={('fixedcard','r0050',e,r) for e in (20,32,64,128,256,512,1024,2048) for r in (1,2,3)}
Path(str(p)+'.audit.json').write_text(json.dumps({'pass':True,'csv_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ids_sha256':hashlib.sha256(Path(str(p)+'.ids.tsv').read_bytes()).hexdigest(),'parameter_rows':24,'query_outputs_checked':24000,'independently_rescored':True,'original_float_predicates_checked':True},indent=2)+'\n');print('PASS replay audit:',p)
