import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90)
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
# precompute per-survey scores dict {(surv,fieldname):score} and group membership
PS={}
for surv in range(1,9):
    off=1+(surv-1)*2
    pages={p:s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True) for p in range(2)}
    for page,img in pages.items():
        Hs,Ws=img.shape[:2]
        Bs=gp._in_region(g.detect_orange_bubbles(img),gp._survey_region(fields,page),Ws,Hs)
        Bb,Wb,Hb=meta[page]; A=gp._icp_affine(Bb[:,:2].copy(),Bs[:,:2].copy())
        for q,opts in groups.items():
            o=[f for f in opts if f.get('page',0)==page]
            if not o: continue
            P=[]
            for f in o:
                c=cal.get(f['name']); P.append(None if (c is None or A is None) else (A@np.array([c[0],c[1],1.0]))[:2])
            valid=[p for p in P if p is not None]; shift=np.zeros(2)
            if valid:
                Vp=np.array(valid); cmin=Vp.min(0)-45;cmax=Vp.max(0)+45
                Bn=Bs[((Bs[:,0]>cmin[0])&(Bs[:,0]<cmax[0])&(Bs[:,1]>cmin[1])&(Bs[:,1]<cmax[1]))]
                if len(Bn)>=len(Vp):
                    from scipy.optimize import linear_sum_assignment
                    best=None
                    for dx in range(-28,29,2):
                        for dy in range(-28,29,2):
                            d=np.sqrt(((Vp+[dx,dy])[:,None,:]-Bn[None,:,:2])**2).sum(2)
                            D=np.sqrt(d); ri,ci=linear_sum_assignment(D); cc=D[ri,ci].mean()
                            if best is None or cc<best[0]: best=(cc,np.array([dx,dy]))
                    shift=best[1]
            for f,p in zip(o,P):
                if p is None: PS[(surv,f['name'])]=0.0; continue
                p=p+shift; dd=np.sqrt((Bs[:,0]-p[0])**2+(Bs[:,1]-p[1])**2); j=int(dd.argmin())
                if dd[j]<=16: cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                else: cx,cy,r=int(p[0]),int(p[1]),13
                PS[(surv,f['name'])]=g.fill_score(img,cx,cy,r)

def evalp(absmin,relf,margin):
    cor=0
    for surv in range(1,9):
        for q,opts in groups.items():
            o=[f for f in opts if (surv,f['name']) in PS]
            if not o: continue
            sc={f['name']:PS[(surv,f['name'])] for f in o}
            if all(v==0 for v in sc.values()): pred=None
            else:
                vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
                med=np.median(list(sc.values()));best=max(sc,key=sc.get)
                ok=(top>=absmin) and ((top>=relf*med if med>0.03 else True) or (top-sec>=margin))
                pred=best.split(': ')[-1] if ok else None
            t=gt.get((surv,q))
            if t is None and q in [x.split('|')[1] for x in []]: pass
            cor+=(pred==t) or (t is None and pred is None)
    return cor
best=None
for absmin in [0.05,0.08,0.10,0.13]:
    for relf in [1.4,1.8,2.2]:
        for margin in [0.03,0.05,0.08]:
            c=evalp(absmin,relf,margin)
            if best is None or c>best[0]: best=(c,absmin,relf,margin)
print('BEST:',best,'=>',f'{100*best[0]/160:.1f}%')
