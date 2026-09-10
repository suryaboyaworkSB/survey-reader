import survey_ocr as s, grid_detect as g, cv2, numpy as np, json, collections
cfg=json.load(open('form_config.json')); fields=cfg['fields']
gcfg=cfg.get('groups',{})
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)

results={}
for surv,off in enumerate([0,2,4,6],1):
    pages={}
    for p in range(2):
        pages[p]=s.pdf_page_to_image('scans/your_scan.pdf',off+p,dpi=300,scan_rotation=90,color=True)
    bubs={p:g.detect_orange_bubbles(pages[p]) for p in pages}
    for q,opts in groups.items():
        p=opts[0].get('page',0)
        if len(opts)<2: continue
        msel=gcfg.get(q,{}).get('max_selections',1)
        res,dbg=g.read_group(pages[p],opts,bubs[p],max_selections=msel)
        yes=[nm.split(': ')[-1] for nm,v in res.items() if v=='Yes']
        sc=dbg['scores']; top=max(sc.values())
        results[(surv,q)]=(yes,top)

# print compact table
qs=[q for q in groups if len(groups[q])>=2]
print(f"{'Question':24} | "+" | ".join(f"S{i}" for i in range(1,5)))
for q in qs:
    cells=[]
    for surv in range(1,5):
        if (surv,q) in results:
            yes,top=results[(surv,q)]
            cells.append((",".join(yes) if yes else "—")[:16]+f"({top:.2f})")
        else: cells.append("?")
    print(f"{q:24} | "+" | ".join(f"{c:22}" for c in cells))
