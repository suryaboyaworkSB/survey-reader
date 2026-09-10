import survey_ocr as s, grid_detect as g, cv2, numpy as np, json
cfg=json.load(open('form_config.json')); fields=cfg['fields']
img=s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True)  # survey1
bub=g.detect_orange_bubbles(img)

def score_variant(color_img,cx,cy,r,erode,thr_mult,cap):
    inner=max(3,int(r)-3); pad=int(r)+5; h,w=color_img.shape[:2]
    x1=max(0,cx-pad);y1=max(0,cy-pad);x2=min(w,cx+pad);y2=min(h,cy+pad)
    roi=color_img[y1:y2,x1:x2]; rh,rw=roi.shape[:2]
    lx=min(max(0,cx-x1),rw-1);ly=min(max(0,cy-y1),rh-1)
    mask=np.zeros((rh,rw),np.uint8); cv2.circle(mask,(lx,ly),inner,255,-1)
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    orange=cv2.inRange(hsv,g._ORANGE_LO,g._ORANGE_HI)
    iv=gray[mask==255]; paper=np.percentile(iv,80); thr=min(paper*thr_mult,cap)
    dark=((gray<thr)&(orange==0)).astype(np.uint8)
    if erode: dark=cv2.erode(dark,np.ones((2,2),np.uint8))
    return int(np.sum((mask==255)&(dark==1)))/max(1,int(np.sum(mask==255)))

tests={'Q3 - Came From':'Work','Q11 - Frequency':'6-11 times/year'}
for q,ans in tests.items():
    opts=[f for f in fields if f['name'].startswith(q) and f.get('page',0)==0]
    grid=g.localize_group(img,opts,bub)
    print(f"\n=== {q}  (true={ans}) ===")
    for variant in [('erode',0.72,135,True),('noErode',0.72,135,False),('noErode',0.85,150,False),('noErode',0.90,170,False)]:
        nm,tm,cap,er=variant
        sc={f['name'].split(': ')[-1]:score_variant(img,*grid[f['name']][:3],er,tm,cap) for f in opts}
        top=sorted(sc.items(),key=lambda x:-x[1])[:3]
        marked=sc.get(ans,-1)
        print(f"  thr_mult={tm} cap={cap} erode={er}: true({ans})={marked:.3f} | top3={[(k[:12],round(v,3)) for k,v in top]}")
