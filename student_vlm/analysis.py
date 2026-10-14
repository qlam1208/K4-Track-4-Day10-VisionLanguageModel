"""Post-hoc analyses of saved raw predictions; no model or TEST-based tuning."""
import numpy as np
from scripts.score import evaluate,parse_output,iou_xyxy
from student_vlm.labels import ground_truth


def paired_f1_interval(entries,raw_a,raw_b,n_boot=1000,seed=42):
    if not entries or n_boot<1:raise ValueError('Need images and a positive bootstrap budget')
    arrays=[]
    for predictions in (raw_a,raw_b):
        rows=[]
        for record in entries:
            m=evaluate([record],{record['image_id']:predictions[record['image_id']]},n_boot=1)['micro']
            rows.append([m['tp'],m['fp'],m['fn']])
        arrays.append(np.asarray(rows,dtype=float))
    index=np.random.default_rng(seed).integers(0,len(entries),(n_boot,len(entries)))
    def f1(x):
        tp,fp,fn=x[...,0],x[...,1],x[...,2]
        denom=2*tp+fp+fn
        return np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0)
    delta=f1(arrays[1][index].sum(1))-f1(arrays[0][index].sum(1))
    return dict(difference_b_minus_a=float(f1(arrays[1].sum(0))-f1(arrays[0].sum(0))),
                ci95=np.quantile(delta,[.025,.975]).tolist(),n_boot=n_boot,unit='paired images',seed=seed)


def error_images(entries,predictions):
    """Image-level mechanical error flags; not a human semantic diagnosis."""
    counts=dict(invalid_json=0,missed_gt=0,wrong_class=0,mislocalized=0,duplicate=0,hallucinated=0)
    for r in entries:
        parsed=parse_output(predictions[r['image_id']])
        if parsed is None:
            counts['invalid_json']+=1
            continue
        gt=ground_truth(r['objects'])
        pairs=sorted(((iou_xyxy(g['bbox'],p['bbox']),gi,pi)
                      for gi,g in enumerate(gt) for pi,p in enumerate(parsed)
                      if g['label']==p['label']),reverse=True)
        used_g,used_p=set(),set()
        for overlap,gi,pi in pairs:
            if overlap<.5:break
            if gi not in used_g and pi not in used_p:used_g.add(gi);used_p.add(pi)
        flags=set()
        if len(used_g)<len(gt):flags.add('missed_gt')
        for pi,p in enumerate(parsed):
            if pi in used_p:continue
            hits=[(iou_xyxy(g['bbox'],p['bbox']),g['label']==p['label']) for g in gt]
            if any(v>=.5 and not same for v,same in hits):flags.add('wrong_class')
            elif any(v>=.5 and same for v,same in hits):flags.add('duplicate')
            elif any(v>0 and same for v,same in hits):flags.add('mislocalized')
            else:flags.add('hallucinated')
        for key in flags:counts[key]+=1
    return counts
