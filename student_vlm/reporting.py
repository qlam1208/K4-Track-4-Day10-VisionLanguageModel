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
    ablations=sorted(Path(evidence_dir).glob('e4*.json'))
    if len(ablations)<2:
        raise FileNotFoundError('Run both rank 4 and rank 64 ablation experiments')
    paths+=ablations
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
    from student_vlm.analysis import paired_f1_interval,error_images
    import numpy as np
    entries=read_jsonl(Path(data_root)/'splits/test.jsonl')
    lookup={r['name'].lower():r for r in rows}
    raw={name:{r['image_id']:r['raw'] for r in read_jsonl(f'work/{name}_test.jsonl')}
         for name in lookup}
    def replace_answer(number,text):
        nonlocal report
        pattern=rf'(### Câu {number}[^\n]*\n\n).*?(?=\n### Câu|\n## Nhìn)'
        report=re.sub(pattern,lambda m:m.group(1)+text+'\n',report,flags=re.S)
    e0=lookup['e0']
    errors=error_images(entries,raw['e0'])
    top=sorted(errors,key=errors.get,reverse=True)[:2]
    first=next(iter(raw['e0'].values()))
    replace_answer(2,f"E0 parse_valid_rate={e0['parse_valid_rate']:.6f}. Hai cờ lỗi tự động phổ biến nhất "
        f"(đếm ảnh, có thể chồng nhau): {top[0]}={errors[top[0]]}, {top[1]}={errors[top[1]]}. "
        "Missed GT dùng ghép cùng lớp IoU≥0,5; wrong_class là FP chồng GT khác lớp ở IoU≥0,5; "
        "mislocalized là FP cùng lớp có giao nhau nhưng chưa khớp. Đây là phân loại hình học, cần đối chiếu ảnh "
        "trước kết luận nguyên nhân ngữ nghĩa. Output thô đầu tiên (giữ nguyên):\n\n```text\n"+first+"\n```")
    wrong=evaluate(entries,raw['e1'],coord='pixel')
    coord_diag=dict(source='same saved E1 raw predictions; post-hoc scale interpretation, no new inference',
                    correct_f1=lookup['e1']['micro']['f1'],wrong_pixel_f1=wrong['micro']['f1'])
    Path('report/diagnostics/coordinate_model_check.json').write_text(json.dumps(coord_diag,indent=2))
    replace_answer(3,f"Trên cùng raw E1, chấm norm1000 có F1={coord_diag['correct_f1']:.6f}; "
        f"đọc nhầm như pixel cho F1={coord_diag['wrong_pixel_f1']:.6f} "
        "(`coordinate_model_check.json`). Đây là chấm lại cùng output để kiểm tra thang, không train/tune lại "
        "bằng TEST. x phải nhân width/1000, y nhân height/1000; ảnh 640×360 dùng hai hệ số khác nhau. "
        "Kiểm tra đơn vị tự tạo riêng `[400,300,600,700]` cũng đã xác nhận đọc sai thang phá IoU.")
    resource='; '.join(f"{n}: F1={lookup[n]['micro']['f1']:.4f}, "
        f"VRAM={lookup[n]['meta']['peak_vram_gb']:.3f} GB, "
        f"train={lookup[n]['meta']['train_minutes']:.2f} phút" for n in ('e1','e2','e3'))
    choice=json.loads(Path('report/diagnostics/ablation_choice.json').read_text())
    replace_answer(7,resource+f". E4 base được chọn là {choice['base']} theo VAL: "
        f"{choice['validation_f1']}. Không chọn base theo TEST. "
        "Đọc cùng CI ở bảng; chênh điểm nhỏ hơn bất định chưa đủ chứng minh lợi ích. "
        "E2 trả thêm gradient/Adam cho ~109M projector; E3 còn cần gradient qua ViT, "
        "nên so F1 với VRAM/thời gian thực đo thay vì chỉ số trainable.")
    intervals='; '.join(f"{r['name']}: [{r['f1_ci95'][0]:.4f}, {r['f1_ci95'][1]:.4f}], "
        f"rộng {r['f1_ci95'][1]-r['f1_ci95'][0]:.4f}" for r in rows)
    replace_answer(8,"CI95 bootstrap theo ảnh từ scorer: "+intervals+
        ". CI chồng nhau không phải phép kiểm định chính thức về hiệu số; có phân tích ghép cặp E4 riêng. "
        "TEST có 311 ảnh nhưng chỉ 3 phiên, tương quan frame có thể làm bootstrap theo ảnh lạc quan. "
        "Cần thêm phiên độc lập trước kết luận tổng quát hóa sang chuyến quay mới.")
    worst=min(('pothole','crack','manhole'),key=lambda c:lookup['e1']['per_class'][c]['f1'])
    cls=lookup['e1']['per_class'][worst]
    areas=[o['yolo'][2]*o['yolo'][3] for r in entries for o in r['objects'] if o['label']==worst]
    replace_answer(9,f"E1 có lớp thấp nhất là {worst}: F1={cls['f1']:.6f}, TP={cls['tp']}, "
        f"FP={cls['fp']}, FN={cls['fn']}. Median diện tích GT lớp đó={np.median(areas):.4%} khung hình. "
        "Ảnh dưới chọn 3 F1 thấp và 3 F1 cao trên cùng E1, GT xanh/pred đỏ; không dùng chúng để chọn lại model. "
        "Crack mảnh có thể nhạy IoU, pothole có thể lẫn bóng, manhole có thể lẫn họa tiết; đó là giả thuyết "
        "cần đọc ảnh, không coi số mẫu là nguyên nhân đã chứng minh.\n\n"+
        '\n'.join(f'![E1 lỗi {i}](report/figures/e1_error_{i:02d}.png)' for i in range(1,7)))
    delta=paired_f1_interval(entries,raw['e4_rank_4'],raw['e4_rank_64'])
    Path('report/diagnostics/e4_paired_ci.json').write_text(json.dumps(delta,indent=2))
    replace_answer(10,f"Base={choice['base']} chọn bằng VAL. r=4 và r=64, alpha=2r, "
        "mọi biến khác giữ nguyên. "
        f"F1 rank4={lookup['e4_rank_4']['micro']['f1']:.6f}, "
        f"rank64={lookup['e4_rank_64']['micro']['f1']:.6f}. "
        f"Hiệu rank64−rank4={delta['difference_b_minus_a']:.6f}; paired CI95={delta['ci95']} "
        "(1.000 bootstrap ảnh cùng chỉ số, seed42). Khoảng chứa 0 chưa cho thấy khác biệt ổn định; "
        "nếu không chứa 0 vẫn chỉ áp dụng trên tập này, với giới hạn tương quan của 3 phiên. Không chọn "
        "rank bằng TEST để train lại.")
    report=report.replace('**Trạng thái: code và kiểm tra CPU đã làm; chưa hoàn thành thí nghiệm GPU E0–E4.**',
                          '**Đã có evidence E0–E4 thực tế; cần đọc ảnh lỗi và rà soát phân tích trước nộp.**')
    pending='Máy chạy hiện tại không có CUDA/NVIDIA driver dùng được.'
    if pending in report:
        start=report.index(pending)
        stop=report.index('## Bảng tổng hợp',start)
        report=report[:start]+"Bảng và các số thực nghiệm được sinh từ scorer gốc và raw đã lưu. " \
            +"Diagnostics kiến trúc/encoding là kiểm tra riêng, không thay thế kết quả GPU.\n\n"+report[stop:]
    report=report.replace('Đã hoàn thành code, dữ liệu QA, encoding và kiểm thử CPU. **Việc còn lại để nộp đạt yêu cầu:** đăng nhập Colab, cấp T4, chạy E0–E3 và hai rank E4, lưu evidence/log/ảnh dự đoán, cập nhật các phân tích thực nghiệm còn thiếu, rồi chạy check_submission.py không còn FAIL.',
        'Đã có code, data QA và kết quả GPU. Trước nộp phải đọc các ảnh lỗi/output để rà soát diễn giải, giữ log nhẹ, và chạy check_submission.py không còn FAIL.')
    report=report.replace('Chưa đo mức ổn định/VRAM trên GPU thật.',
        'Chi phí GPU thực đo nằm ở bảng; loss và gradient norm hữu hạn được kiểm tra mỗi bước và ghi log.')
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
