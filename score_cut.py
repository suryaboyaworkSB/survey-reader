import survey_ocr as s, grid_pipeline as gp, json, collections
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90,overrides=gp.load_overrides('calibration_overrides.json'))
print('calibrated options',sum(1 for v in cal.values() if v),'/',len(cal))
cor=ans=nans=0; perq=collections.defaultdict(lambda:[0,0])
for surv in range(1,9):
    off=1+(surv-1)*2
    pages={p:s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True) for p in range(2)}
    res=gp.read_survey(pages,fields,cal,meta,gcfg)
    for q,pred in res.items():
        if (surv,q) not in gt: continue
        t=gt[(surv,q)]; ok=(pred==t) or (t is None and pred is None)
        cor+=ok; perq[q][0]+=ok; perq[q][1]+=1
        if t: nans+=1; ans+=(pred==t)
print(f"CUT-BLANK PIPELINE overall: {cor}/160={100*cor/160:.1f}%  answered {ans}/{nans}={100*ans/nans:.1f}%")
for q in sorted(perq): print(f"  {q:26} {perq[q][0]}/{perq[q][1]}")
