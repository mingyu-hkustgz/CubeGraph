#!/usr/bin/env python3
"""Small independent integer-vector fixture for the new workload paths."""
import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path
import numpy as np
from convert import columns

def vec(path, a):
    data = np.empty((len(a), a.shape[1]+1), '<f4')
    data[:, 0].view('<i4')[:] = a.shape[1]
    data[:, 1:] = a
    data.tofile(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('output',type=Path);a=ap.parse_args()
    src=Path(__file__).resolve().parent;out=a.output.resolve();fixture=out/'fixture'
    fixture.mkdir(exist_ok=False);data=fixture/'data';ds=data/'sift';ds.mkdir(parents=True)
    rng=np.random.default_rng(20261028);n=12000
    x=rng.integers(0,100,(n,8)).astype('<f4');q=rng.integers(0,100,(1000,8)).astype('<f4')
    vec(ds/'sift_base.fvecs',x);vec(ds/'sift_query.fvecs',q)
    for d in (2,3,4):
        m=rng.random((n,d)).astype('<f4')
        with (ds/f'sift_metadata_uniform_{d}d.bin').open('wb') as f:
            f.write(struct.pack('<QQ',n,d));m.tofile(f)
        native=fixture/f'native{d}';native.mkdir()
        for name,arr in [('vectors.bin',x),('queries.bin',q)]:
            with (native/name).open('wb') as f:f.write(struct.pack('<ii',*arr.shape));arr.tofile(f)
        columns(native/'meta.bin',[f'a{i}' for i in range(d)],m)
        if d==2:
            for i in range(d):np.searchsorted(np.unique(m[:,i]),m[:,i]).astype('<i4').tofile(native/f'attr{i}.bin')
        wl=fixture/f'boxes{d}'
        subprocess.run([out/'bin/generate_boxes','--data-root',data,'--dataset','sift','--metadata',f'uniform_{d}d','--output-dir',wl,'--threads','2','--mode','fixed-cardinality'],check=True)
        path=wl/f'sift_uniform_{d}d_r0050.bin'
        subprocess.run([sys.executable,src/'verify_boxes.py',data,'sift',f'uniform_{d}d',path,'--workers','2'],check=True)
        subprocess.run([sys.executable,src/'convert.py',data,'sift',f'uniform_{d}d',path,native,fixture/f'converted{d}'],check=True)
    radius=fixture/'radius';results=fixture/'radius-results'
    for d,kind in ((2,0),(3,0),(4,0),(2,1)):
        subprocess.run([out/'bin/generate_radius',ds,str(d),str(kind),'1000',radius],check=True,env={**os.environ,'OMP_NUM_THREADS':'2'})
    subprocess.run([sys.executable,src/'verify_radius.py',ds,radius,results,'--gt-only','--workers','2'],check=True)
    print('PREFLIGHT_PASS: 7000 independent GT queries and three native conversions',flush=True)

if __name__=='__main__':main()
