import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank('NJTransitSurvey2026_BLANK.pdf',fields)
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
# preload scans + per-page ICP A + bubbles
DATA={}
for surv in range(1,9):
    off=1+(surv-1)*2
    for p in range(2):
        img=s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True)
        Bs=g.detect_orange_bubbles(img)
        A=gp._icp_affine(meta[p][0][:,:2].copy(),Bs[:,:2].copy())
        DATA[(surv,p)]=(img,Bs,A)

def run(snap,absmin,relf,margin):
    cor=ans=nans=0
    for surv in range(1,9):
        for q,opts in groups.items():
            p=opts[0].get('page',0); img,Bs,A=DATA[(surv,p)]
            o=[f for f in opts if f.get('page',0)==p]
            sc={}
            for f in o:
                c=cal.get(f['name'])
                if c is None or A is None: sc[f['name']]=0.0; continue
                q2=A@np.array([c[0],c[1],1.0])
                dd=np.sqrt((Bs[:,0]-q2[0])**2+(Bs[:,1]-q2[1])**2);j=int(dd.argmin())
                if dd[j]<=snap: cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                else: cx,cy,r=int(q2[0]),int(q2[1]),13
                sc[f['name']]=g.fill_score(img,cx,cy,r)
            if all(v==0 for v in sc.values()): pred=None
            else:
                vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
                med=np.median(list(sc.values()));best=max(sc,key=sc.get)
                ok=(top>=absmin) and ((top>=relf*med if med>0.03 else True) or (top-sec>=margin))
                pred=best.split(': ')[-1] if ok else None
            t=gt.get((surv,q))
            cor+=(pred==t) or (t is None and pred is None)
            if t: nans+=1; ans+=(pred==t)
    return cor,ans,nans

for snap in [16,22,28]:
    for absmin in [0.06,0.10,0.15]:
        for relf in [1.5,2.2]:
            cor,ans,nans=run(snap,absmin,relf,0.05)
            print(f"snap={snap} absmin={absmin} relf={relf}: {cor}/160={100*cor/160:.0f}%  ans {ans}/{nans}={100*ans/nans:.0f}%")
