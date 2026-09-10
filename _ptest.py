import survey_ocr as s, cv2, numpy as np, json, collections
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)

img=s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True)
H,W=img.shape[:2]
bub=s.detect_orange_bubbles(img).astype(float)  # cx,cy,r

def local_fit(opts):
    # config centers
    P=np.array([[ (f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in opts])
    # candidate bubbles in expanded bbox
    x0,y0=P.min(0)-80; x1,y1=P.max(0)+80
    m=(bub[:,0]>x0)&(bub[:,0]<x1)&(bub[:,1]>y0)&(bub[:,1]<y1)
    B=bub[m]
    if len(B)<len(P): return None,None,len(B)
    # search translation by trying each bubble as anchor for centroid alignment
    best=None
    # coarse: align centroids, then refine over small grid
    for tb in [B.mean(0)[:2]-P.mean(0)]:
        for ddx in range(-60,61,6):
            for ddy in range(-60,61,6):
                t=tb+[ddx,ddy]
                Pt=P+t
                # nearest dist sum
                D=np.sqrt(((Pt[:,None,:]-B[None,:,:2])**2).sum(2))
                ri,ci=linear_sum_assignment(D)
                cost=D[ri,ci].mean()
                if best is None or cost<best[0]:
                    best=(cost,t,B,ci)
    return best,P,len(B)

tot=[]
for q,opts in groups.items():
    opts=[f for f in opts if f.get('page',0)==0]
    if len(opts)<2: continue
    best,P,nb=local_fit(opts)
    if best is None:
        print(f"{q:24} only {nb} bubbles for {len(opts)} opts"); continue
    print(f"{q:24} opts={len(opts):2} bubbles_in_region={nb:2} local-assign residual mean={best[0]:.1f}px")
    tot.append(best[0])
print(f"\nMEAN over groups: {np.mean(tot):.1f}px (was ~37px median with global snap)")
