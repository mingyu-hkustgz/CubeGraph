import csv,struct,json,sys,hashlib
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);ds=sys.argv[2];meta=sys.argv[3];wl=Path(sys.argv[4]);data=Path(sys.argv[5]);kind=sys.argv[6]
assert kind == 'cube'
ratios=[100,200,500,1000]
if len(sys.argv)>7:
 assert len(sys.argv)==9 and sys.argv[7]=='--ratio'
 ratios=[int(sys.argv[8])];assert ratios[0] in (50,100,200,500,1000)
rows=list(csv.DictReader(p.open()));assert len(rows)==len(ratios)*(24 if kind=='cube' else 9)
b=(data/ds/f'{ds}_metadata_{meta}.bin').read_bytes();n,d=struct.unpack_from('<QQ',b);attrs=np.frombuffer(b,dtype='<f4',offset=16).reshape(n,d)
keys=set();verified=[];cached={}
for r in rows:
 ratio=round(float(r['target_ratio'])*10000);ef=int(r['budget']);rep=int(r.get('repeat',1));key=(ratio,ef,rep);assert key not in keys;keys.add(key)
 assert int(r['queries'])==1000 and int(r['k'])==20 and float(r['qps'])>0
 if ratio not in cached:
  w=wl/f'{ds}_{meta}_r{ratio:04}.bin';bb=w.read_bytes();nq,wd,k=struct.unpack_from('<QQQ',bb);assert (nq,wd,k)==(1000,d,20)
  boxes=np.frombuffer(bb,dtype='<f4',offset=24,count=nq*d*2).reshape(nq,2,d)
  gt=np.frombuffer(bb,dtype='<i8',offset=24+nq*d*8).reshape(nq,k)
  for qi,row in enumerate(gt):assert np.all((row>=0)&(row<n)) and len(set(row))==20 and np.all((attrs[row]>=boxes[qi,0])&(attrs[row]<=boxes[qi,1]))
  cached[ratio]=(boxes,gt,hashlib.sha256(bb).hexdigest())
 boxes,gt,wh=cached[ratio]
 ids=Path(str(p)+f'.r{ratio}.ef{ef}'+(f'.rep{rep}' if kind=='cube' else '')+'.ids')
 raw=ids.read_bytes();actual=np.frombuffer(raw,dtype='<i8').reshape(1000,20);hits=0
 for qi,row in enumerate(actual):
  assert np.all(row>=-1);row=row[row>=0];assert np.all(row<n) and len(set(row))==len(row)
  assert np.all((attrs[row]>=boxes[qi,0])&(attrs[row]<=boxes[qi,1]))
  hits+=len(set(row)&set(gt[qi]))
 recall=hits/20000;assert abs(recall-float(r['recall']))<1e-8
 verified.append({'ratio':ratio,'ef':ef,'repeat':rep,'recall':recall,'ids_sha256':hashlib.sha256(raw).hexdigest(),'workload_sha256':wh})
expected={(r,e,p) for r in ratios for e in ([20,32,64,128,256,512,1024,2048] if kind=='cube' else [20,32,64,128,256,512,1024,2048,4096]) for p in (range(1,4) if kind=='cube' else [1])};assert keys==expected
Path(str(p)+'.audit.json').write_text(json.dumps({'status':'PASS','csv_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':verified},indent=2)+'\n');print('AUDIT_PASS',p,len(rows))
