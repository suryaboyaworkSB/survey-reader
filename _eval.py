import importlib, survey_ocr as s, grid_detect, cv2, json, numpy as np
importlib.reload(grid_detect); g=grid_detect
cfg=json.load(open('form_config.json')); fields=cfg['fields']
GT={(1,'Q3 - Came From'):'Work',(2,'Q5 - To Bus'):'Walked Only',
    (2,'Q16 - Trip Purpose'):'Work',(2,'Q11 - Frequency'):'5 days/week',
    (2,'Q12 - Ticket'):'Rail Monthly Pass'}
imgs={1:s.pdf_page_to_image('scans/your_scan.pdf',0,dpi=300,scan_rotation=90,color=True),
      2:s.pdf_page_to_image('scans/your_scan.pdf',2,dpi=300,scan_rotation=90,color=True)}
bubs={k:g.detect_orange_bubbles(v) for k,v in imgs.items()}
# global page correction (per page0) from grayscale
corr={}
for k,v in imgs.items():
    gray=cv2.cvtColor(v,cv2.COLOR_BGR2GRAY)
    cb=[f for f in fields if f.get('type')=='checkbox' and f.get('page',0)==0]
    corr[k]=s.estimate_page_correction(gray,cb)
ok=0
for (surv,q),ans in GT.items():
    opts=[f for f in fields if f['name'].startswith(q) and f.get('page',0)==0]
    res,dbg=g.read_group(imgs[surv],opts,bubs[surv],max_selections=1,precorrect=corr[surv])
    yes=[nm.split(': ')[-1] for nm,v in res.items() if v=='Yes']
    pred=yes[0] if yes else None
    sc=dbg['scores']; good=pred==ans; ok+=good
    print(f"{'OK ' if good else 'XX '}S{surv} {q:20} pred={str(pred)[:18]:18} true={ans[:18]:18} | trueScr={sc.get(q+': '+ans,-1):.3f} conf={dbg['confident']}")
print(f"\n{ok}/{len(GT)} verified correct")
