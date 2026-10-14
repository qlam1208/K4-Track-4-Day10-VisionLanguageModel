"""Synthetic fixtures in pytest tmp dirs test the report writer, never produce lab evidence."""
import json
from pathlib import Path
import pytest
from scripts.score import evaluate
from student_vlm.reporting import evidence_rows,update_report

ROOT=Path(__file__).parents[1]


def test_report_requires_both_ablation_runs(tmp_path):
    with pytest.raises(FileNotFoundError):evidence_rows(tmp_path)


def test_report_writer_uses_raw_results_and_is_repeatable(tmp_path,monkeypatch):
    report=(ROOT/'REPORT.md').read_text()
    monkeypatch.chdir(tmp_path)
    Path('REPORT.md').write_text(report)
    for folder in ('data/splits','report/evidence','report/diagnostics','work'):
        Path(folder).mkdir(parents=True,exist_ok=True)
    Path('data/splits/meta.json').write_text(json.dumps({'dataset_version':'test-fixture-only'}))
    entries=[]
    raw={}
    for i,label in enumerate(('pothole','crack','manhole')):
        record=dict(image_id=str(i),width=640,height=360,objects=[dict(label=label,yolo=[.2,.2,.2,.2])])
        entries.append(record)
        raw[str(i)]=json.dumps([dict(label=label,bbox_2d=[100,100,300,300])])
    Path('data/splits/test.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in entries))
    names=['e0','e1','e2','e3','e4_rank_4','e4_rank_64']
    for name in names:
        metric=evaluate(entries,raw,n_boot=2)
        metric.update(name=name.upper(),score_version=1,split='test',dataset_version='test-fixture-only',
                      meta=dict(trainable_params=1,peak_vram_gb=1,train_minutes=1))
        Path(f'report/evidence/{name}.json').write_text(json.dumps(metric))
        Path(f'work/{name}_test.jsonl').write_text(''.join(json.dumps(dict(image_id=k,raw=v))+'\n' for k,v in raw.items()))
    Path('report/diagnostics/ablation_choice.json').write_text(json.dumps(dict(base='E1',validation_f1={'E1':1,'E2':1,'E3':1})))
    update_report()
    first=Path('REPORT.md').read_text()
    assert 'E0 parse_valid_rate=1.000000' in first
    assert 'Chưa chạy E0' not in first
    assert 'F1 rank4=1.000000' in first
    assert 'TEST có 311 ảnh' in first  # The template describes this lab, not a claim about fixture size.
    assert Path('report/diagnostics/e4_paired_ci.json').exists()
    update_report()
    assert Path('REPORT.md').read_text()==first
