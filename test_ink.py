import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np, cv2
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90,overrides=gp.load_overrides('calibration_overrides.json'))

def fill_dark(color_img,cx,cy,r):
    inner=max(3,int(r)-2);pad=int(r)+5;h,w=color_img.shape[:2]
    x1=max(0,cx-pad);y1=max(0,cy-pad);x2=min(w,cx+pad);y2=min(h,cy+pad)
    roi=color_img[y1:y2,x1:x2]
    if roi.size<12: return 0.0
    rh,rw=roi.shape[:2];lx=min(max(0,cx-x1),rw-1);ly=min(max(0,cy-y1),rh-1)
    mask=np.zeros((rh,rw),np.uint8);cv2.circle(mask,(lx,ly),inner,255,-1)
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    orange=cv2.inRange(hsv,g._ORANGE_LO,g._ORANGE_HI)
    iv=gray[mask==255];paper=np.percentile(iv,80);thr=min(paper*0.80,150)
    dark=((gray<thr)&(orange==0)&(mask==255)).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(dark,8)
    keep=np.zeros_like(dark)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=3: keep[lab==i]=1
    # darkness-weighted: sum of (paper-gray) over kept pixels / (paper * total)
    deficit=np.clip(paper-gray.astype(float),0,None)*(keep==1)
    total=int(np.sum(mask==255))
    return float(deficit.sum())/max(1.0,paper*total)

def score(fillfn):
    cor=a=n=0
    for surv in range(1,9):
        off=1+(surv-1)*2
        for page in range(2):
            img=s.pdf_page_to_image(SRC,off+page,dpi=300,scan_rotation=90,color=True);Hs,Ws=img.shape[:2]
            Bs=gp._in_region(g.detect_orange_bubbles(img),gp._survey_region(fields,page),Ws,Hs)
            Bb,Wb,Hb=meta[page];A=gp._icp_affine(Bb[:,:2].copy(),Bs[:,:2].copy())
            groups=collections.OrderedDict()
            for f in fields:
                if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
            for q,opts in groups.items():
                o=[f for f in opts if f.get('page',0)==page]
                if not o: continue
                P=[(A@np.array([cal[f['name']][0],cal[f['name']][1],1.0]))[:2] if cal.get(f['name']) else None for f in o]
                valid=[p for p in P if p is not None];shift=np.zeros(2)
                if valid:
                    Vp=np.array(valid);cmin=Vp.min(0)-45;cmax=Vp.max(0)+45;Bn=Bs[((Bs[:,0]>cmin[0])&(Bs[:,0]<cmax[0])&(Bs[:,1]>cmin[1])&(Bs[:,1]<cmax[1]))]
                    if len(Bn)>=len(Vp):
                        best=None
                        for dx in range(-28,29,2):
                            for dy in range(-28,29,2):
                                D=np.sqrt(((Vp+[dx,dy])[:,None,:]-Bn[None,:,:2])**2).sum(2);ri,ci=linear_sum_assignment(np.sqrt(D));cc=np.sqrt(D)[ri,ci].mean()
                                if best is None or cc<best[0]:best=(cc,np.array([dx,dy]))
                        shift=best[1]
                sc={}
                for f,p in zip(o,P):
                    if p is None: sc[f['name']]=0.0;continue
                    p=p+shift;dd=np.sqrt((Bs[:,0]-p[0])**2+(Bs[:,1]-p[1])**2);j=int(dd.argmin())
                    if dd[j]<=16:cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                    else:cx,cy,r=int(p[0]),int(p[1]),13
                    sc[f['name']]=fillfn(img,cx,cy,r)
                if all(v==0 for v in sc.values()):pred=None
                else:
                    vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
                    med=np.median(list(sc.values()));bestn=max(sc,key=sc.get)
                    ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
                    pred=bestn.split(': ')[-1] if ok else None
                t=gt.get((surv,q))
                if t is None and (surv,q) not in gt: continue
                cor+=(pred==t) or (t is None and pred is None)
                if t: n+=1; a+=(pred==t)
    return cor,a,n
import sys
print('current fill_score:', score(g.fill_score))
print('darkness-weighted :', score(fill_dark))
