import importlib, survey_ocr as s, grid_detect, json, collections, numpy as np
importlib.reload(grid_detect); g=grid_detect
from scipy.optimize import linear_sum_assignment
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}

# preload images+bubbles
DATA={}
for surv in range(1,9):
    off=1+(surv-1)*2
    for p in range(2):
        img=s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True)
        DATA[(surv,p)]=(img,g.detect_orange_bubbles(img))

def localize(img,opts,B,margin,rng,init):
    H,W=img.shape[:2]
    P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in opts],float)
    x0,y0=P.min(0)-margin;x1,y1=P.max(0)+margin;m=(B[:,0]>x0)&(B[:,0]<x1)&(B[:,1]>y0)&(B[:,1]<y1);Bc=B[m]
    if len(Bc)==0: return {f['name']:(int(P[i,0]),int(P[i,1]),13,False) for i,f in enumerate(opts)}
    t0=(Bc[:,:2].mean(0)-P.mean(0)) if init=='centroid' else np.zeros(2)
    best=None
    for dx in range(-rng,rng+1,2):
        for dy in range(-rng,rng+1,2):
            Pt=P+t0+[dx,dy];D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
            if len(Bc)>=len(Pt): ri,ci=linear_sum_assignment(D);c=D[ri,ci].mean()
            else: c=D.min(1).mean()
            if best is None or c<best[0]:best=(c,np.array(t0)+[dx,dy])
    Pt=P+best[1];D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
    if len(Bc)>=len(Pt): ri,ci=linear_sum_assignment(D);idx=ci;dd=D[np.arange(len(Pt)),ci]
    else: idx=D.argmin(1);dd=D[np.arange(len(Pt)),idx]
    out={}
    for i,f in enumerate(opts):
        if dd[i]<=16: b=Bc[int(idx[i])];out[f['name']]=(int(b[0]),int(b[1]),int(round(b[2])),True)
        else: out[f['name']]=(int(Pt[i,0]),int(Pt[i,1]),13,False)
    return out

def score(margin,rng,init):
    cor=ans=nans=0
    for surv in range(1,9):
        for q,opts in groups.items():
            p=opts[0].get('page',0); img,B=DATA[(surv,p)]
            o=[f for f in opts if f.get('page',0)==p]
            grid=localize(img,o,B,margin,rng,init)
            sc={f['name']:(g.fill_score(img,*grid[f['name']][:3]) if grid[f['name']][3] else 0.0) for f in o}
            real=any(grid[f['name']][3] for f in o)
            if not real: pred=None
            else:
                best=max(sc,key=sc.get);vals=sorted(sc.values(),reverse=True);med=np.median(list(sc.values()))
                top=vals[0];sec=vals[1] if len(vals)>1 else 0
                ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
                pred=best.split(': ')[-1] if ok else None
            truth=gt.get((surv,q))
            cor+=(pred==truth) or (truth is None and pred is None)
            if truth: nans+=1; ans+=(pred==truth)
    return cor,ans,nans

for init in ['config','centroid']:
    for margin in [55,75]:
        for rng in [22,40,60]:
            cor,ans,nans=score(margin,rng,init)
            print(f"init={init:8} margin={margin} rng={rng}: overall {cor}/160={100*cor/160:.0f}%  answered {ans}/{nans}={100*ans/nans:.0f}%")
