import survey_ocr as s, grid_detect as g, cv2, json, numpy as np
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']

def localize(img,opts,bub,margin=90):
    H,W=img.shape[:2]
    P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in opts],float)
    x0,y0=P.min(0)-margin; x1,y1=P.max(0)+margin
    m=(bub[:,0]>x0)&(bub[:,0]<x1)&(bub[:,1]>y0)&(bub[:,1]<y1); B=bub[m]
    if len(B)==0: return {f['name']:(int(p[0]),int(p[1]),13,False) for f,p in zip(opts,P)}
    t0=B[:,:2].mean(0)-P.mean(0)
    best=None
    for step,rng in ((6,70),(2,8)):
        base=t0 if best is None else best[1]
        for dx in range(-rng,rng+1,step):
            for dy in range(-rng,rng+1,step):
                t=np.array([base[0]+dx,base[1]+dy]); Pt=P+t
                D=np.sqrt(((Pt[:,None,:]-B[None,:,:2])**2).sum(2))
                if len(B)>=len(Pt):
                    ri,ci=linear_sum_assignment(D); cost=D[ri,ci].mean()
                else: cost=D.min(1).mean()
                if best is None or cost<best[0]: best=(cost,t)
    Pt=P+best[1]; D=np.sqrt(((Pt[:,None,:]-B[None,:,:2])**2).sum(2))
    if len(B)>=len(Pt): ri,ci=linear_sum_assignment(D); dd=D[ri,ci]; idx=ci
    else: idx=D.argmin(1); dd=D[np.arange(len(Pt)),idx]
    out={}
    for i,f in enumerate(opts):
        if dd[i]<=16: b=B[int(idx[i])]; out[f['name']]=(int(b[0]),int(b[1]),int(round(b[2])),True)
        else: out[f['name']]=(int(Pt[i,0]),int(Pt[i,1]),13,False)
    return out

GT={(1,'Q3 - Came From'):'Work',(2,'Q5 - To Bus'):'Walked Only',
    (2,'Q16 - Trip Purpose'):'Work',(2,'Q11 - Frequency'):'5 days/week',
    (2,'Q12 - Ticket'):'Rail Monthly Pass'}
imgs={1:s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True),
      2:s.pdf_page_to_image('scans/your_scan.pdf',2,dpi=300,scan_rotation=90,color=True)}
bubs={k:g.detect_orange_bubbles(v) for k,v in imgs.items()}
ok=0
for (surv,q),ans in GT.items():
    opts=[f for f in fields if f['name'].startswith(q) and f.get('page',0)==0]
    grid=localize(imgs[surv],opts,bubs[surv])
    sc={f['name'].split(': ')[-1]:(g.fill_score(imgs[surv],*grid[f['name']][:3]) if grid[f['name']][3] else 0.0) for f in opts}
    pred=max(sc,key=sc.get); good=pred==ans; ok+=good
    print(f"{'OK ' if good else 'XX '}S{surv} {q:18} pred={pred[:16]:16}({sc[pred]:.2f}) true={ans[:16]:16}({sc.get(ans,-1):.2f})")
print(f"\n{ok}/{len(GT)} winners correct (pre-gate)")
