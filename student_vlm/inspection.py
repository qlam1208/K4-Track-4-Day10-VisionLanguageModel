"""CPU data QA; GT-only figures are explicitly distinct from prediction evidence."""
import json
from collections import Counter
from pathlib import Path
from PIL import ImageDraw,ImageFont
from scripts.labcommon import CLASSES,read_jsonl
from student_vlm.encoding import open_image
from student_vlm.labels import objects_to_answer


def font(size=16):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'):
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


def draw_boxes(image,boxes,color):
    draw=ImageDraw.Draw(image)
    for box in boxes:
        coords=box.get('bbox_2d',box.get('bbox'))
        rect=[coords[0]*image.width/1000,coords[1]*image.height/1000,
              coords[2]*image.width/1000,coords[3]*image.height/1000]
        draw.rectangle(rect,outline=color,width=2)
        draw.text((rect[0],max(0,rect[1]-17)),box['label'],fill=color,font=font(13),
                  stroke_width=1,stroke_fill='black')


def inspect_data(data_root='data'):
    root=Path(data_root)
    splits={s:read_jsonl(root/'splits'/f'{s}.jsonl') for s in ('train','val','test')}
    sessions={s:{r['session'] for r in rows} for s,rows in splits.items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        if sessions[a]&sessions[b]:raise ValueError(f'Session leakage between {a}/{b}')
    meta=json.loads((root/'splits/meta.json').read_text())
    counts={s:dict(images=len(rows),boxes={c:sum(o['label']==c for r in rows for o in r['objects'])
                  for c in CLASSES},sessions=len(sessions[s])) for s,rows in splits.items()}
    out=dict(dataset_version=meta['dataset_version'],counts=counts,session_overlap=False,
             source='scripts/prepare_data.py on the public Kaggle dataset; measured locally',
             figures_kind='ground truth only: not model predictions')
    path=Path('report/diagnostics/dataset.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2))
    # Take examples only from TRAIN for data debugging; reserve TEST for registered evaluation.
    examples=[]
    used=set()
    for label in CLASSES:
        selected=[r for r in splits['train'] if r['image_id'] not in used
                  and any(o['label']==label for o in r['objects'])][:2]
        examples.extend(selected);used.update(r['image_id'] for r in selected)
    for i,r in enumerate(examples):
        image=open_image(r,root)
        boxes=json.loads(objects_to_answer(r['objects']))
        draw_boxes(image,boxes,'lime')
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,25),fill='black')
        draw.text((5,3),f"GT ONLY / TRAIN / {r['image_id']}",fill='white',font=font(12))
        dest=Path('report/figures')/f'gt_train_{i+1:02d}.png'
        dest.parent.mkdir(parents=True,exist_ok=True);image.save(dest)
    return out
