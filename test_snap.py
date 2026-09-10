import survey_ocr as s, grid_detect as g, grid_pipeline as gp, json, collections, numpy as np
from scipy.optimize import linear_sum_assignment
cfg=json.load(open('form_config.json')); fields=cfg['fields']
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90,overrides=gp.load_overrides('calibration_overrides.json'))
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')=='checkbox': groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)

def read(SRC,off_list,snap):
    res={}
    for surv,off in off_list:
        for page in range(2):
            img=s.pdf_page_to_image(SRC,off+page,dpi=300,scan_rotation=90,color=True);Hs,Ws=img.shape[:2]
            Bs=gp._in_region(g.detect_orange_bubbles(img),gp._survey_region(fields,page),Ws,Hs)
            Bb,Wb,Hb=meta[page];A=gp._icp_affine(Bb[:,:2].copy(),Bs[:,:2].copy())
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
                                D=np.sqrt(((Vp+[dx,dy])[:,None,:]-Bn[None,:,:2])**2).sum(2);ri,ci=linear_sum_assignment(np.sqrt(D));c=np.sqrt(D)[ri,ci].mean()
                                if best is None or c<best[0]:best=(c,np.array([dx,dy]))
                        shift=best[1]
                sc={}
                for f,p in zip(o,P):
                    if p is None: sc[f['name']]=0.0;continue
                    p=p+shift;dd=np.sqrt((Bs[:,0]-p[0])**2+(Bs[:,1]-p[1])**2);j=int(dd.argmin())
                    if dd[j]<=snap:
                        cx,cy,r=int(Bs[j,0]),int(Bs[j,1]),int(round(Bs[j,2]))
                        sc[f['name']]=g.fill_score(img,cx,cy,r)
                    else:
                        sc[f['name']]=0.0
                if all(v==0 for v in sc.values()):pred=None
                else:
                    vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
                    med=np.median(list(sc.values()));bestn=max(sc,key=sc.get)
                    ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
                    pred=bestn.split(': ')[-1] if ok else None
                res[(surv,q)]=pred
    return res

gt12={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
d513=json.load(open('answer_key_5_13.json'))['answers']
SRC12='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
SRC13='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_13.pdf'
off12=[(i,1+(i-1)*2) for i in range(1,9)]
serials13=['0480','0603','2011','2580']; off13=[(serials13[i],1+i*2) for i in range(4)]
for snap in [16,22,28,34]:
    r12=read(SRC12,off12,snap); r13=read(SRC13,off13,snap)
    c=n=0
    for (sv,q),t in gt12.items():
        p=r12.get((sv,q));
        if t: n+=1;c+=(p==t)
    c2=n2=0
    for sv in d513:
        for q,t in d513[sv].items():
            p=r13.get((sv,q))
            if t: n2+=1;c2+=(p==t)
    print(f'snap={snap}: 5_12 {c}/{n}={100*c/n:.0f}%  5_13 {c2}/{n2}={100*c2/n2:.0f}%  combined {100*(c+c2)/(n+n2):.0f}%')
