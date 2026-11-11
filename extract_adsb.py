import sys, tarfile, gzip, json, pickle, numpy as np
BOX=(24.0,50.0,-125.5,-66.0)
out={}; n=0; names=[]
tf=tarfile.open(fileobj=sys.stdin.buffer, mode='r|')
for m in tf:
    if not m.isfile(): continue
    n+=1
    if n<=5 or 'trace' not in m.name: names.append(m.name)
    if '/traces/' not in m.name or 'trace_full' not in m.name: continue
    raw=tf.extractfile(m).read()
    try:
        if raw[:2]==b'\x1f\x8b': raw=gzip.decompress(raw)
        j=json.loads(raw)
    except Exception as e:
        continue
    rows=[]
    for p in j.get('trace',[]):
        if p[1] is None or not (BOX[0]<=p[1]<=BOX[1] and BOX[2]<=p[2]<=BOX[3]): continue
        ab=p[3] if isinstance(p[3],(int,float)) else np.nan
        ag=p[10] if len(p)>10 and isinstance(p[10],(int,float)) else np.nan
        if np.isnan(ab) and np.isnan(ag): continue
        rows.append((j['timestamp']+p[0],p[1],p[2],ab,ag,p[4] if isinstance(p[4],(int,float)) else np.nan))
    if len(rows)>=3:
        out[j['icao']]=dict(type=j.get('t',''),reg=j.get('r',''),desc=j.get('desc',''),a=np.array(rows,dtype=np.float64))
    if n%20000==0: print(n,len(out),flush=True)
pickle.dump(out,open('conus_0908.pkl','wb'),protocol=4)
print('done',n,len(out),names[:40],flush=True)
