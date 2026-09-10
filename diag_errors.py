import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90)
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)

# replicate read_survey but capture scores per option
def read_detail(scan_pages):
    out={}
    for page,img in scan_pages.items():
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
                Vp=np.array(valid);cmin=Vp.min(0)-45;cmax=Vp.max(0)+45
                Bn=Bs[((Bs[:,0]>cmin[0])&(Bs[:,0]<cmax[0])&(Bs[:,1]>cmin[1])&(Bs[:,1]<cmax[1]))]
                if len(Bn)>=len(Vp):
                    best=None
                    for dx in range(-28,29,2):
                        for dy in range(-28,29,2):
                            D=np.sqrt(((Vp+[dx,dy])[:,None,:]-Bn[None,:,:2])**2).sum(2)**.5
                            ri,ci=linear_sum_assignment(D);cc=D[ri,ci].mean()
                            if best is None or cc<best[0]:best=(cc,np.array([dx,dy]))
                    shift=best[1]
            sc={}; snapdist={}; calib={}
            for f,p in zip(o,P):
                calib[f['name']]=p is not None
                if p is None: sc[f['name']]=0.0; snapdist[f['name']]=999; continue
                p=p+shift; dd=np.sqrt((Bs[:,0]-p[0])**2+(Bs[:,1]-p[1])**2); j=int(dd.argmin())
                snapdist[f['name']]=float(dd[j])
                if dd[j]<=16: cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                else: cx,cy,r=int(p[0]),int(p[1]),13
                sc[f['name']]=g.fill_score(img,cx,cy,r)
            out[q]=(sc,calib,snapdist)
    return out

cats=collections.Counter()
detail=[]
for surv in range(1,9):
    off=1+(surv-1)*2
    pages={p:s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True) for p in range(2)}
    det=read_detail(pages)
    for q,opts in groups.items():
        if (surv,q) not in gt: continue
        truth=gt[(surv,q)]
        sc,calib,snap=det.get(q,({},{},{}))
        if not sc: continue
        # current decision
        if all(v==0 for v in sc.values()): pred=None
        else:
            vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
            med=np.median(list(sc.values()));bestn=max(sc,key=sc.get)
            ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
            pred=bestn.split(': ')[-1] if ok else None
        if (pred==truth) or (truth is None and pred is None): continue  # correct
        # categorize the error
        tkey=q+': '+truth if truth else None
        tscore=sc.get(tkey,None) if truth else None
        tcal=calib.get(tkey,None) if truth else None
        if truth is None:
            cat='FALSE_POSITIVE (truth blank, predicted '+str(pred)+')'
        elif tcal is False or tkey not in sc:
            cat='CALIB_GAP (truth option not calibrated)'
        elif tscore is not None and tscore<0.06:
            cat='MISLOCATED (truth bubble scored ~0 -> wrong position/faint)'
        elif tscore is not None and pred is None:
            cat='UNDER_GATE (truth had ink but gated to blank)'
        else:
            cat='WRONG_WINNER (other bubble scored higher)'
        cats[cat.split(' ')[0]]+=1
        detail.append((surv,q,truth,pred,round(tscore,3) if tscore is not None else None,tcal))

print('=== ERROR CATEGORIES ===')
for k,v in cats.most_common(): print(f'  {k:14} {v}')
print(f'  total errors: {sum(cats.values())}')
print('\n=== sample of each ===')
seen=set()
for surv,q,truth,pred,ts,tc in detail:
    key=q.split(' ')[0]
print('\n=== all errors (survey,q,truth,pred,truthScore,truthCalibrated) ===')
for d in detail: print('  ',d)
