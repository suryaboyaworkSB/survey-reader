"""Blank-anchored bubble reading: calibrate option->bubble once on the clean
template, then globally register that template onto each scan."""
import survey_ocr as s, grid_detect as g, numpy as np, json, collections, os
import cv2
from scipy.optimize import linear_sum_assignment

def _survey_region(fields, page, pad_frac=0.12):
    """Fractional bounding box (x0,y0,x1,y1) of this page's option bubbles.
    Used to discard detections outside the survey area (e.g. the promo/Win
    block and reply-mail barcode on the back page have their own circles)."""
    pos=[(f["x"]+f["w"]/2, f["y"]+f["h"]/2) for f in fields
         if f.get("type")=="checkbox" and f.get("page",0)==page]
    if not pos: return (0.0,0.0,1.0,1.0)
    xs=[p[0] for p in pos]; ys=[p[1] for p in pos]
    return (min(xs)-pad_frac, min(ys)-pad_frac,
            max(xs)+pad_frac, max(ys)+pad_frac)

def _in_region(B, region, W, H):
    x0,y0,x1,y1=region
    m=((B[:,0]>=x0*W)&(B[:,0]<=x1*W)&(B[:,1]>=y0*H)&(B[:,1]<=y1*H))
    return B[m]

def _icp_affine(P, Q, iters=40, keep=0.6):
    if len(P)<3 or len(Q)<3: return None
    A=np.array([[Q[:,0].std()/max(P[:,0].std(),1),0,0],[0,Q[:,1].std()/max(P[:,1].std(),1),0]],float)
    A[0,2]=Q[:,0].mean()-A[0,0]*P[:,0].mean(); A[1,2]=Q[:,1].mean()-A[1,1]*P[:,1].mean()
    for _ in range(iters):
        Pt=(A@np.c_[P,np.ones(len(P))].T).T
        D=np.sqrt(((Pt[:,None,:]-Q[None,:,:])**2).sum(2)); idx=D.argmin(1); d=D[np.arange(len(P)),idx]
        m=d<np.percentile(d,keep*100)
        if m.sum()<3: break
        A=np.linalg.lstsq(np.c_[P[m],np.ones(m.sum())],Q[idx[m]],rcond=None)[0].T
    return A


_ORB=None
_ORB_SCALE=0.6      # detect features on a downscaled image (~3x faster, same H)
def _orb():
    global _ORB
    if _ORB is None: _ORB=cv2.ORB_create(nfeatures=2500)
    return _ORB

def _detect(img_bgr):
    """ORB on a downscaled grayscale image. Returns (keypoints, descriptors)
    with keypoint coordinates already scaled back to full resolution."""
    g=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    small=cv2.resize(g,None,fx=_ORB_SCALE,fy=_ORB_SCALE,interpolation=cv2.INTER_AREA)
    kp,des=_orb().detectAndCompute(small,None)
    if kp is not None:
        inv=1.0/_ORB_SCALE
        for k in kp: k.pt=(k.pt[0]*inv, k.pt[1]*inv)
    return kp,des

def blank_features(blank_img):
    """Precompute ORB keypoints/descriptors for a blank page (done once)."""
    try:
        return _detect(blank_img)
    except Exception:
        return (None,None)

def fit_homography(blank_feats, scan_img):
    """Register blank->scan with ORB features + RANSAC homography.

    Features come from the PRINTED form (question text, rule lines, logo), which
    a respondent's pen never alters — so alignment stays accurate even right next
    to a heavily marked bubble whose orange ring is broken and undetectable.
    blank_feats = (keypoints, descriptors) precomputed for the blank page.
    Returns a 3x3 homography (blank px -> scan px), or None if matching is weak.
    """
    try:
        k1,d1=blank_feats
        if d1 is None: return None
        k2,d2=_detect(scan_img)
        if d2 is None: return None
        bf=cv2.BFMatcher(cv2.NORM_HAMMING)
        good=[m for m,n in bf.knnMatch(d1,d2,k=2) if m.distance<0.75*n.distance]
        if len(good)<25: return None
        src=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)  # blank
        dst=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)  # scan
        H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5.0)
        if H is None or mask.sum()<20: return None
        return H
    except Exception:
        return None

def _poly_feats(P, W, H):
    """Quadratic basis on coordinates normalised to ~[0,1] for stability."""
    x=P[:,0]/W; y=P[:,1]/H
    return np.c_[np.ones(len(P)), x, y, x*x, y*y, x*y]

def fit_warp(Bb, Bs, W, H, iters=20, keep=0.65):
    """Fit a smooth quadratic warp blank->scan from bubble correspondences.

    A single affine can't follow the page's non-uniform distortion (the cut
    leaves a smooth gradient), which throws a whole column ~a row off where its
    own bubbles are marked/undetected. A quadratic warp fit from ALL reliable
    correspondences models that gradient page-wide, so every option maps
    accurately even when its own bubble wasn't detected. Returns warp(P)->P'.
    """
    A=_icp_affine(Bb[:,:2].copy(), Bs[:,:2].copy())
    if A is None: return None
    coef=None
    Bbxy=Bb[:,:2]; Bsxy=Bs[:,:2]
    Pt=(A@np.c_[Bbxy,np.ones(len(Bbxy))].T).T
    for _ in range(iters):
        D=np.sqrt(((Pt[:,None,:]-Bsxy[None,:,:])**2).sum(2))
        idx=D.argmin(1); d=D[np.arange(len(Bbxy)),idx]
        m=d<max(8.0, np.percentile(d,keep*100))
        if m.sum()<8: break
        F=_poly_feats(Bbxy[m],W,H)
        coef=np.linalg.lstsq(F, Bsxy[idx[m]], rcond=None)[0]   # 6x2
        Pt=_poly_feats(Bbxy,W,H)@coef
    if coef is None: return None
    def warp(P):
        P=np.atleast_2d(np.asarray(P,float))
        return _poly_feats(P,W,H)@coef
    return warp

def calibrate_blank(blank_pdf, fields, scan_rotation=0, overrides=None):
    """Return {field_name: (cx,cy)} in blank px, per page, + blank bubbles/dims.

    Per-group calibration: for each question, translate the config option
    centres onto the detected blank bubbles (search + Hungarian assignment).
    Use scan_rotation=90 when the blank is a rotated portrait scan (same as the
    filled forms), 0 for an upright landscape template.
    """
    groups=collections.OrderedDict()
    for f in fields:
        if f.get('type')!='checkbox': continue
        groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
    cal={}; meta={}
    for page in sorted({f.get('page',0) for f in fields if f.get('type')=='checkbox'}):
        img=s.pdf_page_to_image(blank_pdf,page,dpi=300,scan_rotation=scan_rotation,color=True)
        H,W=img.shape[:2]
        B=_in_region(g.detect_orange_bubbles(img), _survey_region(fields,page), W, H)
        meta[page]=(B,W,H,img,blank_features(img))
        # Global pre-alignment: the blank's scan framing (margins, cut strip) can
        # differ from the config's coordinate frame by a large offset/scale.
        # Fit a robust global affine (config -> blank bubbles) ONCE per page so
        # the per-group search starts already roughly aligned. Without this, a
        # differently-framed blank sits hundreds of px from the config and the
        # small per-group search can never reach it.
        page_opts=[f for f in fields
                   if f.get('type')=='checkbox' and f.get('page',0)==page]
        Pall=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H]
                       for f in page_opts],float)
        def _med_resid(Q):
            D=np.sqrt(((Q[:,None,:]-B[None,:,:2])**2).sum(2))
            return float(np.median(D.min(1)))
        # Global pre-alignment maps the config onto this blank's framing (the
        # front page sits ~200px out). A few groups whose config is already
        # accurate can be distorted by it; those are fixed by manual overrides
        # in the locked calibration rather than by changing this global step.
        Aglob=_icp_affine(Pall.copy(), B[:,:2].copy(), iters=60, keep=0.55)
        def _pre(P):
            if Aglob is None: return P
            return (Aglob@np.c_[P,np.ones(len(P))].T).T
        for q,opts in groups.items():
            o=[f for f in opts if f.get('page',0)==page]
            if not o: continue
            Praw=np.array([[(f['x']+f['w']/2)*W,(f['y']+f['h']/2)*H]
                           for f in o],float)
            for f in o: cal[f['name']]=None
            fit=_fit_group_to_bubbles(_pre(Praw), B, len(o))
            if fit is None: continue
            Bc, ri, ci, D = fit
            md=float(np.median(D[ri,ci])) if len(ri) else 0.0
            for r,c2 in zip(ri,ci):
                if D[r,c2] <= max(40.0, md*2.5):
                    cal[o[r]['name']]=(float(Bc[c2,0]),float(Bc[c2,1]))
    # Manual overrides for groups the auto-calibration gets wrong (verified by
    # eye on the blank). Stored as fractional blank coords {field: [xf,yf]} so
    # they're resolution-independent. They win over the computed positions.
    if overrides:
        page_dims={p:(meta[p][1],meta[p][2]) for p in meta}
        fpage={f["name"]:f.get("page",0) for f in fields}
        for name,(xf,yf) in overrides.items():
            pg=fpage.get(name,0)
            if pg in page_dims:
                W,H=page_dims[pg]; cal[name]=(xf*W, yf*H)
    return cal, meta


def load_overrides(path):
    import os
    if path and os.path.isfile(path):
        return json.load(open(path))
    return None

def _fit_group_to_bubbles(P, B, nopt):
    """Fit option positions P onto detected bubbles B (coarse translation +
    clamped affine ICP). Returns (Bc, row_idx, col_idx, Dist) or None."""
    for margin in (45,70,110,180):
        x0,y0=P.min(0)-margin;x1,y1=P.max(0)+margin
        m=(B[:,0]>x0)&(B[:,0]<x1)&(B[:,1]>y0)&(B[:,1]<y1);Bc=B[m]
        if len(Bc)>=nopt: break
    if len(Bc) < max(1, nopt-3): return None
    best=None
    for dx in range(-60,61,3):
        for dy in range(-60,61,3):
            Pt=P+[dx,dy];D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
            ri,ci=linear_sum_assignment(D);c=D[ri,ci].mean()
            if best is None or c<best[0]:best=(c,np.array([dx,dy]))
    Pt=P+best[1]
    for _ in range(6):
        D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
        ri,ci=linear_sum_assignment(D)
        if len(ri)<3: break
        Aff=np.linalg.lstsq(np.c_[Pt[ri],np.ones(len(ri))],Bc[ci,:2],rcond=None)[0].T
        if not (0.7<abs(Aff[0,0])<1.4 and 0.7<abs(Aff[1,1])<1.4
                and abs(Aff[0,1])<0.3 and abs(Aff[1,0])<0.3):
            break
        Pt=(Aff@np.c_[Pt,np.ones(len(Pt))].T).T
    D=np.sqrt(((Pt[:,None,:]-Bc[None,:,:2])**2).sum(2))
    ri,ci=linear_sum_assignment(D)
    return (Bc, ri, ci, D)

def read_survey(scan_pages, fields, cal, meta, gcfg, annotate_dir=None, label=None,
                debug_scores=None):
    """scan_pages: {page:color_img}. Returns ({group: selected_or_None}, {group: confident}).

    If annotate_dir is given, also saves one overlay image per page showing,
    for every option, the circle the code actually measured: GREEN = selected,
    RED = considered but not selected, plus faint yellow on every detected
    bubble. Lets you eyeball alignment and selection against the real scan.
    """
    groups=collections.OrderedDict()
    for f in fields:
        if f.get('type')!='checkbox': continue
        groups.setdefault(f['name'].split(': ',1)[0],[]).append(f)
    out={}; conf={}
    for page,img in scan_pages.items():
        Hs,Ws=img.shape[:2]
        Bs=_in_region(g.detect_orange_bubbles(img), _survey_region(fields,page), Ws, Hs)
        Bb,Wb,Hb,blank_img,blank_feats=meta[page]
        # Primary alignment: ORB feature homography from the printed form
        # (mark-independent). Falls back to bubble-based warp/affine only if ORB
        # matching is too weak.
        Hom=fit_homography(blank_feats, img)
        # Template-subtracted "respondent ink" image: warp the blank onto the
        # scan and subtract, leaving only what the respondent added (ring/text
        # cancel). Lets faint ticks stand out against a clean background. Needs
        # the accurate ORB homography, so only built when Hom succeeds.
        diff_img=None
        if Hom is not None:
            bw=cv2.warpPerspective(blank_img, Hom, (Ws,Hs))
            bg=cv2.cvtColor(bw,cv2.COLOR_BGR2GRAY).astype(np.int16)
            sg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.int16)
            diff_img=np.clip(bg-sg,0,255).astype(np.uint8)
        A=_icp_affine(Bb[:,:2].copy(), Bs[:,:2].copy())
        warp=fit_warp(Bb, Bs, Ws, Hs) if len(Bs)>=8 else None
        def _map(c):                       # blank px -> scan px
            if Hom is not None:
                p=cv2.perspectiveTransform(
                    np.float32([[[c[0],c[1]]]]), Hom)
                return p[0,0]
            if warp is not None: return warp([c[0],c[1]])[0]
            if A is not None:    return (A@np.array([c[0],c[1],1.0]))[:2]
            return np.array([c[0],c[1]],float)
        draw_items=[]   # (cx,cy,r,is_winner,short_label) for the overlay
        for q,opts in groups.items():
            o=[f for f in opts if f.get('page',0)==page]
            if not o: continue
            # Initial map via the smooth page-wide WARP (handles the non-uniform
            # cut gradient better than a single affine). The per-group refine
            # below then locally perfects it.
            P=[( _map(cal[f['name']]) if cal.get(f['name']) is not None else None)
               for f in o]
            valid=[p for p in P if p is not None]
            valid_idx=[i for i,p in enumerate(P) if p is not None]
            Bn=Bs; Pv=np.array(valid) if valid else np.empty((0,2))
            if valid:
                Vp=np.array(valid)
                cmin=Vp.min(0)-95; cmax=Vp.max(0)+95
                msk=((Bs[:,0]>cmin[0])&(Bs[:,0]<cmax[0])&
                     (Bs[:,1]>cmin[1])&(Bs[:,1]<cmax[1]))
                Bn=Bs[msk]
                shift=np.zeros(2)
                if len(Bn)>=len(Vp):
                    LAM=0.02       # anti-alias: prefer the config-anchored row
                    def _cost(dx, dy):
                        diff=(Vp+[dx,dy])[:,None,:]-Bn[None,:,:2]
                        D=np.sqrt((diff**2).sum(2))
                        ri,ci=linear_sum_assignment(D)
                        return D[ri,ci].mean() + LAM*(abs(dx)+abs(dy))
                    best=None
                    for dx in range(-60,61,8):
                        for dy in range(-60,61,8):
                            c=_cost(dx,dy)
                            if best is None or c<best[0]: best=(c,(dx,dy))
                    cx0,cy0=best[1]
                    for dx in range(cx0-7,cx0+8,2):
                        for dy in range(cy0-7,cy0+8,2):
                            c=_cost(dx,dy)
                            if c<best[0]: best=(c,(dx,dy))
                    shift=np.array(best[1],float)
                Pv=Vp+shift
                if len(Bn)>=len(Pv) and len(Pv)>=5:
                    for _ in range(5):
                        D=np.sqrt(((Pv[:,None,:]-Bn[None,:,:2])**2).sum(2))
                        idx=D.argmin(1); d=D[np.arange(len(Pv)),idx]
                        inl=d<=max(18.0, np.percentile(d,60))
                        if inl.sum()<max(4,int(0.55*len(Pv))): break
                        M=np.linalg.lstsq(np.c_[Vp[inl],np.ones(int(inl.sum()))],
                                          Bn[idx[inl],:2],rcond=None)[0]
                        if not (0.8<M[0,0]<1.25 and 0.8<M[1,1]<1.25
                                and abs(M[1,0])<0.25 and abs(M[0,1])<0.25):
                            break
                        Pv=np.c_[Vp,np.ones(len(Vp))]@M
            sc={f['name']:0.0 for f in o}; pos_r={}
            optp=[(o[valid_idx[k]]['name'], Pv[k]) for k in range(len(valid_idx))]
            cand=Bn if (len(Bn)>0) else Bs
            if optp and len(cand)>0:
                Pp=np.array([p for _,p in optp])
                D=np.sqrt(((Pp[:,None,:]-cand[None,:,:2])**2).sum(2))
                ri,ci=linear_sum_assignment(D)
                amap={int(r):int(c) for r,c in zip(ri,ci) if D[r,c]<=34}
                for idx,(nm,pp) in enumerate(optp):
                    if idx in amap:
                        b=cand[amap[idx]]; cx,cy,r=int(b[0]),int(b[1]),int(round(b[2])) or 13
                    else:
                        cx,cy,r=int(pp[0]),int(pp[1]),13
                    pos_r[nm]=(cx,cy,r)
                    sc[nm]=(g.fill_score_diff(diff_img,cx,cy,r) if diff_img is not None
                            else g.fill_score(img,cx,cy,r))
            winner=None
            if not all(v==0.0 for v in sc.values()):
                vals=sorted(sc.values(),reverse=True);top=vals[0];sec=vals[1] if len(vals)>1 else 0
                med=np.median(list(sc.values()));best=max(sc,key=sc.get)
                ok=(top>=0.10) and ((top>=1.8*med if med>0.03 else True) or (top-sec>=0.05))
                out[q]=best.split(': ')[-1] if ok else None
                winner=best if ok else None
            if debug_scores is not None:
                debug_scores[q]={nm.split(': ')[-1]:
                                 (round(v,3), pos_r.get(nm)) for nm,v in sc.items()}
            # record overlay markers for this group
            for nm,(cx,cy,r) in pos_r.items():
                draw_items.append((cx,cy,r, nm==winner, nm.split(': ')[-1][:14]))
            if all(v==0.0 for v in sc.values()):
                # nothing measured — but a faint mark may simply have been
                # mislocated, so don't blindly trust a blank: flag it.
                out[q]=None; conf[q]=False; continue
            # Confidence: on these faint-pencil scans, correct and incorrect
            # reads have nearly identical ink scores (the errors are mostly
            # registration shifts, not weak marks), so only a clearly dark mark
            # with a wide margin can be trusted. Everything else — faint marks,
            # close calls, and blanks that might be a missed mark — is flagged
            # for review. This deliberately flags most answers; it is honest
            # about how much of this data needs a human glance.
            # Only a clearly dark mark with a wide margin is trusted; faint picks
            # and blanks (possible missed marks) are flagged for review.
            conf[q]= bool(ok and top>=0.50 and (top-sec)>=0.25)

        # ── save alignment overlay for this page ──
        if annotate_dir:
            vis=img.copy()
            for b in Bs:                                   # every detected bubble
                cv2.circle(vis,(int(b[0]),int(b[1])),int(b[2]),(0,200,255),1)
            for cx,cy,r,is_win,lab in draw_items:
                col=(0,200,0) if is_win else (0,0,235)
                cv2.circle(vis,(cx,cy),max(r,10),col,3 if is_win else 2)
                if is_win:
                    cv2.putText(vis,lab,(cx+r+3,cy+5),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,150,0),2)
            os.makedirs(annotate_dir,exist_ok=True)
            fn=os.path.join(annotate_dir,
                            f"{label or 'survey'}_page{page+1}.png")
            cv2.imwrite(fn,vis)
    return out, conf
