import importlib, survey_ocr as s, grid_detect, json, collections, numpy as np
importlib.reload(grid_detect); g=grid_detect
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
groups=collections.OrderedDict()
for f in fields:
    if f.get('type')!='checkbox': continue
    groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}

results={}
for surv in range(1,9):
    off=1+(surv-1)*2
    pg={p:s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True) for p in range(2)}
    bub={p:g.detect_orange_bubbles(pg[p]) for p in pg}
    for q,opts in groups.items():
        p=opts[0].get('page',0); msel=gcfg.get(q,{}).get('max_selections',1)
        res,dbg=g.read_group(pg[p],opts,bub[p],max_selections=msel)
        yes=[nm.split(': ')[-1] for nm,v in res.items() if v=='Yes']
        results[(surv,q)]=(yes[0] if yes else None, dbg['confident'])

cor=0; perq=collections.defaultdict(lambda:[0,0]); blank_ok=0; ans_ok=0; nans=0
for (surv,q),truth in gt.items():
    pred,conf=results.get((surv,q),(None,False))
    ok=(pred==truth) or (truth is None and pred is None)
    cor+=ok; perq[q][0]+=ok; perq[q][1]+=1
    if truth: nans+=1; ans_ok+=(pred==truth)
print(f"NEW overall: {cor}/{len(gt)} = {100*cor/len(gt):.1f}%   answered: {ans_ok}/{nans} = {100*ans_ok/nans:.1f}%")
print("per-question:")
for q in groups:
    if perq[q][1]: print(f"  {q:26} {perq[q][0]}/{perq[q][1]}")
