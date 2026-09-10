import fitz,cv2,numpy as np,json
import survey_ocr as s

cfg=json.load(open('form_config.json'))
fields=cfg['fields'] if 'fields' in cfg else cfg.get('checkboxes',[])
# find field list key
for k in cfg:
    if isinstance(cfg[k],list) and cfg[k] and isinstance(cfg[k][0],dict) and 'x' in cfg[k][0]:
        fields=cfg[k]; print('fields key:',k,len(fields)); break

img=s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True)
h,w=img.shape[:2]
print('page',w,'x',h,'color_scan',s.is_color_scan(img))

# replicate runtime correction
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); cb0=[f for f in fields if f.get("type")=="checkbox" and f.get("page",0)==0]; corr=s.estimate_page_correction(gray, cb0)
print('correction type', type(corr))

cb=[f for f in fields if f.get('type')=='checkbox' and f.get('page',0)==0]
print('checkbox fields page0:',len(cb))
bubbles=s.detect_orange_bubbles(img)
print('orange bubbles',len(bubbles))

vis=img.copy()
grid=s.build_color_grid(img, cb, correction=corr)
for f in cb:
    cx,cy,r=grid[f['name']]
    ex=int((f['x']+f['w']/2)*w); ey=int((f['y']+f['h']/2)*h)
    sc=s._color_ink_density(img,cx,cy,r)
    cv2.circle(vis,(ex,ey),3,(0,0,255),-1)        # red=raw expected
    cv2.line(vis,(ex,ey),(cx,cy),(255,0,255),1)
    cv2.circle(vis,(cx,cy),r,(0,255,0),2)         # green=snapped
    cv2.putText(vis,f"{sc:.2f}",(cx-14,cy-r-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),1)
cv2.imwrite('/sessions/wonderful-elegant-newton/mnt/outputs/viz_p0.png',vis)
print('saved')
