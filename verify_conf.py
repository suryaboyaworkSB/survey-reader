import survey_ocr as s, grid_pipeline as gp, json, collections
cfg=json.load(open('form_config.json')); fields=cfg['fields']; gcfg=cfg.get('groups',{})
BLANK='/sessions/wonderful-elegant-newton/mnt/uploads/BRNB422005F2960_004268.pdf'
SRC='/sessions/wonderful-elegant-newton/mnt/uploads/Mailback 5_12-18c66566.pdf'
gt={tuple([int(k.split('|')[0]),k.split('|',1)[1]]):v for k,v in json.load(open('/sessions/wonderful-elegant-newton/mnt/outputs/ground_truth.json')).items()}
cal,meta=gp.calibrate_blank(BLANK,fields,scan_rotation=90,overrides=gp.load_overrides('calibration_overrides.json'))
# confusion: confident-correct, confident-wrong, review-correct, review-wrong
cc=cw=rc=rw=0; tot=0
for surv in range(1,9):
    off=1+(surv-1)*2
    pages={p:s.pdf_page_to_image(SRC,off+p,dpi=300,scan_rotation=90,color=True) for p in range(2)}
    winners,conf=gp.read_survey(pages,fields,cal,meta,gcfg)
    for q,pred in winners.items():
        if (surv,q) not in gt: continue
        t=gt[(surv,q)]; ok=(pred==t) or (t is None and pred is None); isconf=conf.get(q,True)
        tot+=1
        if isconf and ok: cc+=1
        elif isconf and not ok: cw+=1
        elif not isconf and ok: rc+=1
        else: rw+=1
print(f'Total answers: {tot}')
print(f'CONFIDENT: {cc+cw}  ->  correct {cc} ({100*cc/max(1,cc+cw):.0f}%), wrong {cw}')
print(f'NEEDS REVIEW: {rc+rw}  ->  correct {rc}, wrong {rw}')
print(f'Of all {cw+rw} errors, {rw} ({100*rw/max(1,cw+rw):.0f}%) are flagged for review')
print(f'Accuracy if you trust confident + fix review: {100*(cc+rc+rw)/tot:.0f}% achievable with review of {rc+rw} cells')
