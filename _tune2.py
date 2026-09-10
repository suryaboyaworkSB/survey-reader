import survey_ocr as s, grid_detect as g, cv2, numpy as np, json
cfg=json.load(open('form_config.json')); fields=cfg['fields']

def fscore(color_img,cx,cy,r):
    inner=max(3,int(r)-2); pad=int(r)+5; h,w=color_img.shape[:2]
    x1=max(0,cx-pad);y1=max(0,cy-pad);x2=min(w,cx+pad);y2=min(h,cy+pad)
    roi=color_img[y1:y2,x1:x2]
    if roi.size<12: return 0.0
    rh,rw=roi.shape[:2]; lx=min(max(0,cx-x1),rw-1);ly=min(max(0,cy-y1),rh-1)
    mask=np.zeros((rh,rw),np.uint8); cv2.circle(mask,(lx,ly),inner,255,-1)
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    orange=cv2.inRange(hsv,g._ORANGE_LO,g._ORANGE_HI)
    iv=gray[mask==255]; paper=np.percentile(iv,80); thr=min(paper*0.80,150)
    dark=((gray<thr)&(orange==0)&(mask==255)).astype(np.uint8)
    # remove tiny specks via connected components
    n,lab,stats,_=cv2.connectedComponentsWithStats(dark,8)
    clean=np.zeros_like(dark)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=3: clean[lab==i]=1
    return int(clean.sum())/max(1,int((mask==255).sum()))

def decide(opts,grid,img):
    sc={f['name'].split(': ')[-1]:fscore(img,*grid[f['name']][:3]) for f in opts}
    real=any(grid[f['name']][3] for f in opts)
    if not real: return None,sc
    vals=sorted(sc.values(),reverse=True); top=vals[0]; sec=vals[1] if len(vals)>1 else 0
    med=np.median(list(sc.values()))
    best=max(sc,key=sc.get)
    ok = top>=0.10 and (top>=1.6*med if med>0.02 else True) and (top-sec)>=0.04
    return (best if ok else None), sc

# verified ground truth
GT={(1,'Q3 - Came From'):'Work',(2,'Q5 - To Bus'):'Walked Only',
    (2,'Q16 - Trip Purpose'):'Work',(2,'Q11 - Frequency'):'5 days/week',
    (2,'Q12 - Ticket'):'Rail Monthly Pass'}
imgs={surv:s.pdf_page_to_image('scans/your_scan.pdf',off,dpi=300,scan_rotation=90,color=True) for surv,off in [(1,0),(2,2)]}
bubs={k:g.detect_orange_bubbles(v) for k,v in imgs.items()}
for (surv,q),ans in GT.items():
    opts=[f for f in fields if f['name'].startswith(q) and f.get('page',0)==0]
    grid=g.localize_group(imgs[surv],opts,bubs[surv])
    pred,sc=decide(opts,grid,imgs[surv])
    mark='OK ' if pred==ans else 'XX '
    print(f"{mark}S{surv} {q:20} pred={str(pred)[:18]:18} true={ans[:18]:18} | trueScore={sc.get(ans,-1):.3f} predScore={sc.get(pred,0) if pred else 0:.3f}")
