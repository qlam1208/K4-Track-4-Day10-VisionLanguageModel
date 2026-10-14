"""Grounding labels, prompts, and the optional verifiable IoU reward."""
import json
import math
from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output


def prompt_for(coord='norm1000'):
    if coord not in ('norm1000', 'pixel'):
        raise ValueError(f'Unknown coordinate scale: {coord}')
    scale = ('integer coordinates from 0 to 1000, normalized independently by image width and height'
             if coord == 'norm1000' else 'integer pixel coordinates in the original image')
    return ('Locate every visible road defect: pothole, crack, and manhole. '
            'Return only a JSON list, without Markdown or explanation. Each object must have '
            '"label" (one of pothole, crack, manhole) and "bbox_2d": [x1,y1,x2,y2]. '
            f'Use {scale}; x is horizontal and y is vertical. '
            'Use the tightest box enclosing each defect, with x1<x2 and y1<y2. '
            'Return [] if no defect is visible.')


PROMPT = prompt_for()


def yolo_to_box(label, yolo, *, coord='norm1000', width=640, height=360):
    if label not in CLASSES:
        raise ValueError(f'Unknown class: {label}')
    if len(yolo) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in yolo):
        raise ValueError('YOLO must contain four finite normalized coordinates')
    if coord not in ('norm1000', 'pixel'):
        raise ValueError(f'Unknown coordinate scale: {coord}')
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == 'norm1000' else (width, height)
    box = [round(min(1, max(0, v))*s) for v, s in
           zip((xc-w/2, yc-h/2, xc+w/2, yc+h/2), (sx, sy, sx, sy))]
    return {'label': label, 'bbox_2d': box}


def objects_to_answer(objects, **kwargs):
    boxes = [yolo_to_box(o['label'], o['yolo'], **kwargs) for o in objects]
    boxes.sort(key=lambda b: (b['bbox_2d'][1], b['bbox_2d'][0], b['label']))
    return json.dumps(boxes, separators=(',', ':'))


def ground_truth(objects):
    out = []
    for obj in objects:
        xc, yc, w, h = obj['yolo']
        out.append({'label': obj['label'], 'bbox': tuple(1000*v for v in
                   (xc-w/2, yc-h/2, xc+w/2, yc+h/2))})
    return out


def iou_reward(completion, gt_boxes):
    """GUIDE reward: valid JSON floor, same-class greedy matching, IoU >= .5."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g['bbox'], p['bbox']), gi, pi)
                    for gi, g in enumerate(gt_boxes) for pi, p in enumerate(pred)
                    if g['label'] == p['label']), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for overlap, gi, pi in pairs:
        if overlap < .5:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            total += overlap
    return .2 + .8*total/max(len(gt_boxes), len(pred), 1)
