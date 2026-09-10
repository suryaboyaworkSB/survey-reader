import survey_ocr as s, grid_detect as g, grid_pipeline as gp, cv2, numpy as np, json, collections
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
imgs={p:s.pdf_page_to_image(BLANK,p,dpi=300,scan_rotation=90,color=True) for p in [0,1]}
B={p:gp._in_region(g.detect_orange_bubbles(imgs[p]),gp._survey_region(fields,p),imgs[p].shape[1],imgs[p].shape[0]) for p in [0,1]}
# pre-align per page
A={}
for p in [0,1]:
    po=[f for f in fields if f.get('type')=='checkbox' and f.get('page',0)==p]
    P=np.array([[(f['x']+f['w']/2)*imgs[p].shape[1],(f['y']+f['h']/2)*imgs[p].shape[0]] for f in po])
    raw=np.median(np.sqrt(((P[:,None,:]-B[p][None,:,:2])**2).sum(2)).min(1))
    A[p]=gp._icp_affine(P.copy(),B[p][:,:2].copy(),60,0.55) if raw>50 else None

def assign(group):
    o=[f for f in fields if f['name'].startswith(group) and f.get('type')=='checkbox']
    p=o[0].get('page',0); W,H=imgs[p].shape[1],imgs[p].shape[0]
    P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in o],float)
    if A[p] is not None: P=(A[p]@np.c_[P,np.ones(len(P))].T).T
    # local translation + nearest, one-to-one
    Bc=B[p]
    best=None
    for dx in range(-70,71,3):
        for dy in range(-70,71,3):
            D=np.sqrt(((P+[dx,dy])[:,None,:]-Bc[None,:,:2])**2).sum(2)
            ri,ci=linear_sum_assignment(np.sqrt(D)); c=np.sqrt(D)[ri,ci].mean()
            if best is None or c<best[0]: best=(c,np.array([dx,dy]))
    Pt=P+best[1]; D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
    ri,ci=linear_sum_assignment(D)
    out={}
    for r,c2 in zip(ri,ci): out[o[r]['name']]=(float(Bc[c2,0]),float(Bc[c2,1]))
    return out,p,W,H,o

# render verification for the broken groups
for group in ['Q20 - Traveling With Child','Q24 - Race','Q15 - Return','Q8 - After Bus']:
    ov,p,W,H,o=assign(group)
    vis=imgs[p].copy()
    for b in B[p]: cv2.circle(vis,(int(b[0]),int(b[1])),int(b[2]),(0,160,255),1)
    xs=[];ys=[]
    for f in o:
        c=ov.get(f['name'])
        if c: cv2.circle(vis,(int(c[0]),int(c[1])),15,(0,0,255),2);cv2.putText(vis,f['name'].split(': ')[-1][:11],(int(c[0])+15,int(c[1])+3),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1);xs.append(c[0]);ys.append(c[1])
    x0=int(min(xs)-50);x1=int(max(xs)+360);y0=int(min(ys)-40);y1=int(max(ys)+40)
    crop=vis[max(0,y0):y1,max(0,x0):x1]
    fn=group.split(' ')[0].lower()
    cv2.imwrite(f'/sessions/wonderful-elegant-newton/mnt/outputs/ov_{fn}.png',cv2.resize(crop,(int(crop.shape[1]*1.3),int(crop.shape[0]*1.3))))
    print(f'{group}: rendered ov_{fn}.png ({len(o)} options)')
