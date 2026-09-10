import survey_ocr as s, cv2, numpy as np, json
cfg=json.load(open('form_config.json')); fields=cfg['fields']
img=s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True)
h,w=img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cb=[f for f in fields if f.get('type')=='checkbox' and f.get('page',0)==0]
corr=s.estimate_page_correction(gray, cb)
bub=s.detect_orange_bubbles(img)
ax,bx,ay,by=corr
# residual: each option's corrected-expected -> nearest detected orange bubble
res=[]
snapped_ids=[]
for f in cb:
    ex=(f['x']+f['w']/2)*w; ey=(f['y']+f['h']/2)*h
    cex=ax*ex+bx; cey=ay*ey+by
    d=np.sqrt((bub[:,0]-cex)**2+(bub[:,1]-cey)**2)
    i=int(np.argmin(d)); res.append(d[i]); snapped_ids.append(i)
res=np.array(res)
print(f"options={len(cb)} detected_bubbles={len(bub)}")
print(f"nearest-bubble dist: median={np.median(res):.1f} mean={res.mean():.1f} p90={np.percentile(res,90):.1f} max={res.max():.1f}")
print(f"options within 12px of a bubble: {(res<12).sum()}  within 22px: {(res<22).sum()}  beyond 40px: {(res>40).sum()}")
# how many distinct bubbles claimed (collisions)
import collections
c=collections.Counter(snapped_ids)
dup=sum(1 for k,v in c.items() if v>1)
print(f"distinct bubbles claimed={len(c)} ; bubbles claimed by >1 option={dup}")
# systematic offset of nearest matches (only confident <15px to estimate true bias)
dx=[];dy=[]
for f in cb:
    ex=(f['x']+f['w']/2)*w; ey=(f['y']+f['h']/2)*h
    cex=ax*ex+bx; cey=ay*ey+by
    d=np.sqrt((bub[:,0]-cex)**2+(bub[:,1]-cey)**2); i=int(np.argmin(d))
    if d[i]<15: dx.append(bub[i,0]-cex); dy.append(bub[i,1]-cey)
print(f"residual offset (confident matches n={len(dx)}): dx_med={np.median(dx):.1f} dy_med={np.median(dy):.1f}")
