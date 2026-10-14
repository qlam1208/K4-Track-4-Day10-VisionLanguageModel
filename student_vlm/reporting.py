"""Generate result tables and raw-error panels only from genuine scorer outputs."""
import json
import re
from pathlib import Path
from PIL import ImageDraw
from scripts.labcommon import read_jsonl
from scripts.score import parse_output, evaluate
from student_vlm.encoding import open_image
from student_vlm.inspection import draw_boxes,font
from student_vlm.labels import objects_to_answer

START='<!-- RESULTS_START -->'
END='<!-- RESULTS_END -->'


def evidence_rows(evidence_dir='report/evidence'):
    paths=[Path(evidence_dir)/f'e{i}.json' for i in range(4)]
    paths+=sorted(Path(evidence_dir).glob('e4*.json'))
    if not paths or any(not p.exists() for p in paths):
        raise FileNotFoundError('Run genuine E0–E4 experiments before building the results table')
    rows=[json.loads(p.read_text()) for p in paths]
    versions={r['dataset_version'] for r in rows}
    if len(versions) != 1 or any(r['score_version']!=1 or r['split']!='test'
                                or r['missing_predictions'] for r in rows):
        raise ValueError('Final evidence has inconsistent versions, splits or missing predictions')
    return rows


def update_report(data_root='data'):
    rows=evidence_rows()
    actual=json.loads((Path(data_root)/'splits/meta.json').read_text())['dataset_version']
    if {r['dataset_version'] for r in rows}!={actual}:raise ValueError('Dataset version mismatch')
    table=['| Mã | Trainable | VRAM (GB) | Phút train | F1 | CI95 | FP/ảnh | Pothole | Crack | Manhole |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        m=r['meta']
        values=[r['name'],str(m['trainable_params']),f"{m['peak_vram_gb']:.3f}",
                f"{m['train_minutes']:.2f}",f"{r['micro']['f1']:.4f}",
                f"[{r['f1_ci95'][0]:.4f}, {r['f1_ci95'][1]:.4f}]",f"{r['fp_per_image']:.3f}"]
        values += [f"{r['per_class'][c]['f1']:.4f}" for c in ('pothole','crack','manhole')]
        table.append('| '+' | '.join(values)+' |')
    summary={'dataset_version':actual,'experiments':rows,'kind':'official scorer test results'}
    Path('report/diagnostics/result_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    report=Path('REPORT.md').read_text()
    if START not in report or END not in report:raise ValueError('Results table markers are missing from REPORT.md')
    report=re.sub(re.escape(START)+r'.*?'+re.escape(END),START+'\n'+'\n'.join(table)+'\n'+END,
                  report,flags=re.S)
    # Required empirical answers have an explicit marker; replace them using the measured outputs.
    lookup={r['name'].lower():r for r in rows}
    e0=lookup['e0']
    raw=read_jsonl('work/e0_test.jsonl')
    invalid=next((r['raw'] for r in raw if parse_output(r['raw']) is None),None)
    example=invalid if invalid is not None else raw[0]['raw']
    replacement=(f"E0 parse_valid_rate = {e0['parse_valid_rate']:.6f}. Một output thô thực tế: "
                 f"`{example}`. Phải đọc thêm ảnh lỗi để phân loại lỗi vị trí/lớp; JSON hợp lệ không đảm bảo đúng box.")
    report=report.replace('<!-- E0_RESULT -->',replacement)
    intervals='; '.join(f"{r['name']}: [{r['f1_ci95'][0]:.4f}, {r['f1_ci95'][1]:.4f}]" for r in rows)
    report=report.replace('<!-- CI_RESULT -->',f'Khoảng thực đo: {intervals}.')
    Path('REPORT.md').write_text(report)
    return rows


def prediction_figures(data_root='data'):
    rows=evidence_rows()
    # Fixed E1 for error analysis; never choose the model because of its TEST F1.
    entries=read_jsonl(Path(data_root)/'splits/test.jsonl')
    predictions={r['image_id']:r['raw'] for r in read_jsonl('work/e1_test.jsonl')}
    examples=[]
    for record in entries:
        metric=evaluate([record],{record['image_id']:predictions[record['image_id']]},n_boot=1)
        examples.append((metric['micro']['f1'],record))
    examples.sort(key=lambda x:(x[0],x[1]['image_id']))
    chosen=examples[:3]+examples[-3:]
    captions=[]
    for i,(f1,r) in enumerate(chosen,1):
        image=open_image(r,data_root)
        draw_boxes(image,json.loads(objects_to_answer(r['objects'])),'lime')
        parsed=parse_output(predictions[r['image_id']])
        if parsed is not None:draw_boxes(image,parsed,'red')
        draw=ImageDraw.Draw(image)
        draw.rectangle((0,0,image.width,26),fill='black')
        caption=f"E1 / green=GT, red=pred / F1={f1:.3f} / {r['image_id']}"
        draw.text((4,3),caption,fill='white',font=font(12))
        dest=Path('report/figures')/f'e1_error_{i:02d}.png';image.save(dest)
        captions.append(dict(path=str(dest),image_id=r['image_id'],f1=f1,raw=predictions[r['image_id']]))
    Path('report/figures/captions.json').write_text(json.dumps(captions,ensure_ascii=False,indent=2))
