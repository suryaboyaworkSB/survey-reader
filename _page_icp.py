import survey_ocr as s, grid_detect as g, cv2, json, numpy as np
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']

def page_affine(P, B, iters=25):
    # robust ICP affine P(config)->B(bubbles)
    A=np.array([[1,0,B[:,0].mean()-P[:,0].mean()],[0,1,B[:,1].mean()-P[:,1].mean()]],float)
    for _ in range(iters):
        Pt=(A@np.c_[P,np.ones(len(P))].T).T
        D=np.sqrt(((Pt[:,None,:]-B[None,:,:2])**2).sum(2))
        idx=D.argmin(1); d=D[np.arange(len(P)),idx]
        good=d<np.percentile(d,70)  # reject worst 30% as outliers
        X=np.c_[P[good],np.ones(good.sum())]
        A=np.linalg.lstsq(X,B[idx[good],:2],rcond=None)[0].T
    Pt=(A@np.c_[P,np.ones(len(P))].T).T
    D=np.sqrt(((Pt[:,None,:]-B[None,:,:2])**2).sum(2)); d=D.min(1)
    return A,Pt,d

GT={(1,'Q3 - Came From'):'Work',(2,'Q5 - To Bus'):'Walked Only',
    (2,'Q16 - Trip Purpose'):'Work',(2,'Q11 - Frequency'):'5 days/week',
    (2,'Q12 - Ticket'):'Rail Monthly Pass'}
imgs={1:s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True),
      2:s.pdf_page_to_image('scans/your_scan.pdf',2,dpi=300,scan_rotation=90,color=True)}
for surv in (1,2):
    img=imgs[surv]; H,W=img.shape[:2]
    B=g.detect_orange_bubbles(img)
    cb=[f for f in fields if f.get('type')=='checkbox' and f.get('page',0)==0]
    P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in cb],float)
    A,Pt,d=page_affine(P,B)
    print(f"S{surv}: page-ICP residual median={np.median(d):.1f} p90={np.percentile(d,90):.1f}")
    pos={f['name']:Pt[i] for i,f in enumerate(cb)}
    for (sv,q),ans in GT.items():
        if sv!=surv: continue
        opts=[f for f in cb if f['name'].startswith(q)]
        sc={}
        for f in opts:
            px,py=pos[f['name']]
            # snap to nearest bubble within 14px else use predicted
            dd=np.sqrt((B[:,0]-px)**2+(B[:,1]-py)**2); j=dd.argmin()
            if dd[j]<=14: cx,cy,r=int(B[j,0]),int(B[j,1]),int(round(B[j,2]))
            else: cx,cy,r=int(px),int(py),13
            sc[f['name'].split(': ')[-1]]=g.fill_score(img,cx,cy,r)
        pred=max(sc,key=sc.get)
        print(f"   {'OK ' if pred==ans else 'XX '}{q:18} pred={pred[:16]:16}({sc[pred]:.2f}) true={ans[:16]:16}({sc.get(ans,-1):.2f})")
