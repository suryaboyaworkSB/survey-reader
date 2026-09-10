import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90,overrides=gp.load_overrides('calibration_overrides.json'))
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
rows=[]  # (correct?, predBlank?, top, margin)
for surv in range(1,9):
    off=1+(surv-1)*2
    for page in range(2):
        img=s.pdf_page_to_image(SRC,off+page,dpi=300,scan_rotation=90,color=True);Hs,Ws=img.shape[:2]
        Bs=gp._in_region(g.detect_orange_bubbles(img),gp._survey_region(fields,page),Ws,Hs)
        Bb,Wb,Hb=meta[page];A=gp._icp_affine(Bb[:,:2].copy(),Bs[:,:2].copy())
        for q,opts in groups.items():
            o=[f for f in opts if f.get('page',0)==page]
            if not o or (surv,q) not in gt: continue
            P=[(A@np.array([cal[f['name']][0],cal[f['name']][1],1.0]))[:2] if cal.get(f['name']) else None for f in o]
            valid=[p for p in P if p is not None];shift=np.zeros(2)
            if valid:
                Vp=np.array(valid);cmin=Vp.min(0)-45;cmax=Vp.max(0)+45;Bn=Bs[((Bs[:,0]>cmin[0])&(Bs[:,0]<cmax[0])&(Bs[:,1]>cmin[1])&(Bs[:,1]<cmax[1]))]
                if len(Bn)>=len(Vp):
                    best=None
                    for dx in range(-28,29,2):
                        for dy in range(-28,29,2):
                            D=np.sqrt(((Vp+[dx,dy])[:,None,:]-Bn[None,:,:2])**2).sum(2);ri,ci=linear_sum_assignment(np.sqrt(D));c=np.sqrt(D)[ri,ci].mean()
                            if best is None or c<best[0]:best=(c,np.array([dx,dy]))
                    shift=best[1]
            sc={}
            for f,p in zip(o,P):
                if p is None: sc[f['name']]=0.0;continue
                p=p+shift;dd=np.sqrt((Bs[:,0]-p[0])**2+(Bs[:,1]-p[1])**2);j=int(dd.argmin())
                if dd[j]<=16:cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                else:cx,cy,r=int(p[0]),int(p[1]),13
                sc[f['name']]=g.fill_score(img,cx,cy,r)
            vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
            med=np.median(list(sc.values()));bestn=max(sc,key=sc.get)
            ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
            pred=bestn.split(': ')[-1] if ok else None
            t=gt[(surv,q)]; correct=(pred==t) or (t is None and pred is None)
            rows.append((correct, pred is None, top, top-sec))
import numpy as np
R=np.array([(int(a),int(b),c,d) for a,b,c,d in rows])
corr=R[R[:,0]==1]; wr=R[R[:,0]==0]
print(f'CORRECT picks (non-blank): top median={np.median([r[2] for r in rows if r[0] and not r[1]]):.2f}')
print(f'WRONG picks (non-blank):   top median={np.median([r[2] for r in rows if not r[0] and not r[1]]):.2f}')
# try threshold: confident if non-blank AND top>=X AND margin>=Y
for X,Y in [(0.4,0.15),(0.45,0.2),(0.5,0.25),(0.55,0.3)]:
    confmask=[(not b) and c>=X and d>=Y for a,b,c,d in rows]
    cc=sum(1 for i,(a,b,c,d) in enumerate(rows) if confmask[i] and a)
    cw=sum(1 for i,(a,b,c,d) in enumerate(rows) if confmask[i] and not a)
    print(f'  conf(top>={X},margin>={Y}): {cc+cw} confident, {100*cc/max(1,cc+cw):.0f}% correct')
