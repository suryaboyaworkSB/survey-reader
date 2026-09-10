import survey_ocr as s, cv2, numpy as np, json, collections, sys
cfg=json.load(open('form_config.json')); fields=cfg['fields']
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    pre=f['name'].split(': ',1)[0]
    groups.setdefault(pre,[]).append(f)

def make(surv,off,page,qlist,outname):
    img=s.pdf_page_to_image('scans/your_scan.pdf',off+page,dpi=300,scan_rotation=90,color=True)
    H,W=img.shape[:2]
    rows=[]
    for q in qlist:
        gf=groups[q]
        gf=[f for f in gf if f.get('page',0)==page]
        if not gf: continue
        xs=[(f['x'])*W for f in gf]+[(f['x']+f['w'])*W for f in gf]
        ys=[(f['y'])*H for f in gf]+[(f['y']+f['h'])*H for f in gf]
        x0=max(0,int(min(xs))-30); x1=min(W,int(max(xs))+520)
        y0=max(0,int(min(ys))-25); y1=min(H,int(max(ys))+35)
        crop=img[y0:y1,x0:x1].copy()
        # label band
        band=np.full((34,crop.shape[1],3),255,np.uint8)
        cv2.putText(band,q,(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,200),2)
        rows.append(np.vstack([band,crop]))
    # pad to same width
    maxw=max(r.shape[1] for r in rows)
    rows=[np.hstack([r,np.full((r.shape[0],maxw-r.shape[1],3),255,np.uint8)]) for r in rows]
    sheet=np.vstack(rows)
    # scale so width ~1400
    sc=1400/sheet.shape[1]
    sheet=cv2.resize(sheet,(int(sheet.shape[1]*sc),int(sheet.shape[0]*sc)))
    cv2.imwrite(f'/sessions/wonderful-elegant-newton/mnt/outputs/{outname}.png',sheet)
    print(outname, sheet.shape)

p0q=['Q2 - Board Time','Q3 - Came From','Q5 - To Bus','Q8 - After Bus','Q10 - Going To','Q11 - Frequency','Q12 - Ticket','Q13 - Pay','Q14 - Buy Ticket','Q15 - Return','Q16 - Trip Purpose','Q17 - Alt Mode','Q18 - Gender']
surv=int(sys.argv[1]); off=[0,2,4,6][surv-1]
make(surv,off,0,p0q,f'sheet_s{surv}_p0')
