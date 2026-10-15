"""Keep the finest L layers, preserving native graphs and geometric resolutions."""
import struct,json,hashlib,sys
from pathlib import Path
src=Path(sys.argv[1]).resolve();dest=Path(sys.argv[2]);dest.mkdir(parents=True,exist_ok=True)
b=src.read_bytes();pos=0
def u():
 global pos
 v=struct.unpack_from('<Q',b,pos)[0];pos+=8;return v
def skip(n):
 global pos
 pos+=n;assert pos<=len(b)
head=[u() for _ in range(8)];L,M,ef,cross,d,vd,n,minimum=head
assert L==6 and d==2 and M==16 and cross==2
bd=u();assert bd==d;skip(8*d)
mn,md=u(),u();assert (mn,md)==(n,d);skip(mn*md*4)
prefix=b[:pos];chunks=[];geometries=[]
for j in range(L):
 start=pos;lid,cp,tc=u(),u(),u();assert lid==j and cp==2**(j+1) and tc==cp**d
 skip(d*4);adj=u();assert adj==tc
 for c in range(tc):skip(u()*4)
 bd=u();assert bd==d;skip(8*d)
 chunks.append(b[start:pos]);geometries.append({'native_layer':lid,'cubes_per_dim':cp})
assert pos==len(b)
manifest=[]
for keep in range(1,L+1):
 p=dest/f'L{keep}.cube';data=struct.pack('<Q',keep)+prefix[8:]+b''.join(chunks[L-keep:])
 if keep==L:assert data==b
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 files=[]
 for i,j in enumerate(range(L-keep,L)):
  target=Path(str(src)+f'.layer_{j}');link=Path(str(p)+f'.layer_{i}')
  assert target.is_file()
  if link.exists():assert link.resolve()==target
  else:link.symlink_to(target)
  files.append({'native_layer':j,'source':str(target),'bytes':target.stat().st_size})
 manifest.append({'L':keep,'geometry':geometries[L-keep:],'index_bytes':len(data)+sum(x['bytes'] for x in files),'files':files,'directory_sha256':hashlib.sha256(data).hexdigest()})
(dest/'manifest.json').write_text(json.dumps({'source':str(src),'original_header':head,'subsets':manifest},indent=2)+'\n')
print('SLICE_PASS',src)
