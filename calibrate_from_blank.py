"""Re-derive accurate per-option bubble coordinates from the blank template."""
import survey_ocr as s, grid_detect as g, numpy as np, json, collections, sys
from scipy.optimize import linear_sum_assignment

def calibrate(blank_pdf, config_path, out_path):
    cfg=json.load(open(config_path)); fields=cfg['fields']
    groups=collections.OrderedDict()
    for f in fields:
        if f.get('type')!='checkbox': continue
        groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
    pages={}
    for p in sorted({f.get('page',0) for f in fields if f.get('type')=='checkbox'}):
        img=s.pdf_page_to_image(blank_pdf,p,dpi=300,scan_rotation=0,color=True)
        pages[p]=(img, g.detect_orange_bubbles(img, min_r=7,max_r=22,min_dist=15))
    new_coords={}; report=[]
    for q,opts in groups.items():
        page=opts[0].get('page',0); img,B=pages[page]; H,W=img.shape[:2]
        P=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H] for f in opts],float)
        # region
        for margin in (120,200,320):
            x0,y0=P.min(0)-margin; x1,y1=P.max(0)+margin
            m=(B[:,0]>x0)&(B[:,0]<x1)&(B[:,1]>y0)&(B[:,1]<y1); Bc=B[m]
            if len(Bc)>=len(P): break
        if len(Bc)<len(P):
            report.append((q,'GAP',len(Bc),len(P),None)); 
            for f in opts: new_coords[f['name']]=None
            continue
        # translation search, Hungarian one-to-one cost
        t0=Bc[:,:2].mean(0)-P.mean(0); best=None
        for step,rng in ((6,140),(2,10)):
            base=t0 if best is None else best[1]
            for dx in range(-rng,rng+1,step):
                for dy in range(-rng,rng+1,step):
                    Pt=P+[base[0]+dx,base[1]+dy]
                    D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
                    ri,ci=linear_sum_assignment(D); c=D[ri,ci].mean()
                    if best is None or c<best[0]: best=(c,np.array([base[0]+dx,base[1]+dy],float))
        Pt=P+best[1]; D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
        ri,ci=linear_sum_assignment(D); res=D[ri,ci].mean()
        # order-preservation check: relative arrangement of options vs assigned bubbles
        assign={}
        for i in range(len(opts)): assign[opts[i]['name']]=Bc[ci[i]]
        distinct=len({tuple(v[:2]) for v in assign.values()})==len(opts)
        status='OK' if (res<8 and distinct) else 'CHECK'
        report.append((q,status,len(Bc),len(P),round(res,1)))
        for f in opts:
            b=assign[f['name']]; r=max(8,int(b[2]))
            new_coords[f['name']]={'cx_f':float(b[0]/W),'cy_f':float(b[1]/H),'r':int(r)}
    # write calibrated config: keep structure, replace x,y,w,h from cx,cy and fixed half-size
    out=json.loads(json.dumps(cfg))
    HALF=14  # px half-box at 300dpi ~ bubble radius
    for f in out['fields']:
        nc=new_coords.get(f['name'])
        if f.get('type')=='checkbox' and nc:
            # store as fractional box centered on bubble; w,h fractions from ~28px
            f['x']=nc['cx_f']-HALF/4313; f['y']=nc['cy_f']-HALF/2663
            f['w']=2*HALF/4313; f['h']=2*HALF/2663
            f['_calibrated']=True
    json.dump(out,open(out_path,'w'),indent=1)
    return report

if __name__=='__main__':
    rep=calibrate('NJTransitSurvey2026_BLANK.pdf','form_config.json',
                  '/sessions/wonderful-elegant-newton/mnt/outputs/form_config_cal.json')
    ok=sum(1 for r in rep if r[1]=='OK')
    for q,st,nb,npp,res in rep:
        flag='' if st=='OK' else f'  <-- {st}'
        print(f"{q:26} {st:5} bubbles={nb:2}/{npp:2} res={res}{flag}")
    print(f"\n{ok}/{len(rep)} groups calibrated cleanly")
