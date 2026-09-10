import survey_ocr as s, grid_detect as g, numpy as np, json, collections
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)

def calibrate_group(opts,B,W,H,margin=85):
    P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in opts],float)
    x0,y0=P.min(0)-margin; x1,y1=P.max(0)+margin
    m=(B[:,0]>x0)&(B[:,0]<x1)&(B[:,1]>y0)&(B[:,1]<y1); Bc=B[m]
    if len(Bc)<len(P): return None,len(Bc),None
    t0=Bc[:,:2].mean(0)-P.mean(0); best=None
    for step,rng in ((5,75),(1,7)):
        base=t0 if best is None else best[1]
        for dx in range(-rng,rng+1,step):
            for dy in range(-rng,rng+1,step):
                Pt=P+[base[0]+dx,base[1]+dy]
                D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
                ri,ci=linear_sum_assignment(D); c=D[ri,ci].mean()
                if best is None or c<best[0]: best=(c,np.array([base[0]+dx,base[1]+dy]))
    Pt=P+best[1]; D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
    ri,ci=linear_sum_assignment(D)
    assign={opts[i]['name']:Bc[ci[i]] for i in range(len(opts))}
    distinct=len(set(tuple(v[:2]) for v in assign.values()))==len(opts)
    return best[0],len(Bc),distinct

for page in [0,1]:
    img=s.pdf_page_to_image('NJTransitSurvey2026_BLANK.pdf',page,dpi=300,scan_rotation=0,color=True)
    H,W=img.shape[:2]; B=g.detect_orange_bubbles(img)
    print(f"--- PAGE {page}  ({W}x{H}, {len(B)} bubbles) ---")
    for q,opts in groups.items():
        o=[f for f in opts if f.get('page',0)==page]
        if len(o)<1: continue
        res,nb,distinct=calibrate_group(o,B,W,H)
        if res is None: print(f"  {q:24} opts={len(o):2} ONLY {nb} bubbles in region  <-- gap"); continue
        flag='' if (res<8 and distinct) else '  <-- CHECK'
        print(f"  {q:24} opts={len(o):2} residual={res:4.1f}px distinct={distinct}{flag}")
