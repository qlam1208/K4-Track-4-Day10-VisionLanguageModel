"""Val selects E4's base; test scoring is performed once per registered experiment."""
import copy
import gc
import json
import subprocess
import sys
from pathlib import Path
import yaml
from scripts.labcommon import read_jsonl
from student_vlm.inference import predict
from student_vlm.modeling import load_base, load_finetuned
from student_vlm.training import seed_everything, train


def config_for(name):
    path=Path('configs')/(name.lower()+'.yaml')
    return yaml.safe_load(path.read_text())


def score_predictions(prediction_path,name,split='test',coord='norm1000',meta=None,data_root='data'):
    target=Path('report/evidence' if split=='test' else 'work/validation')/(name.lower()+'.json')
    if target.exists():
        raise FileExistsError(f'{target} already exists; inference/scoring may not silently retest')
    command=[sys.executable,'scripts/score.py','--pred',str(prediction_path),'--name',name,
             '--split',split,'--coord',coord,'--out',str(target),
             '--splits-dir',str(Path(data_root)/'splits')]
    for key,value in (meta or {}).items():
        command += ['--meta',f'{key}={value}']
    subprocess.run(command,check=True)
    return json.loads(target.read_text())


def evaluate_checkpoint(config,checkpoint=None,split='test',data_root='data',limit=None):
    import torch
    name=config['name']
    seed_everything(config.get('seed',42))
    if checkpoint:
        model,processor,run=load_finetuned(checkpoint)
        metadata={k:run[k] for k in ('trainable_params','peak_vram_gb','train_minutes')}
        metadata.update(step=run['step'],rank=config['lora']['r'],setup=config['lora']['targets'])
    else:
        model,processor=load_base(config['model_id'])
        metadata=dict(trainable_params=0,train_minutes=0)
    records=read_jsonl(Path(data_root)/'splits'/f'{split}.jsonl')
    if limit:
        if split=='test':
            raise ValueError('Final test evidence must cover the full split')
        records=records[:limit]
    suffix=f'_{split}_debug_{limit}' if limit else f'_{split}'
    path=Path('work')/(name.lower()+suffix+'.jsonl')
    timing=predict(model,processor,records,path,data_root=data_root,
                   coord=config.get('coord','norm1000'),max_new_tokens=config.get('max_new_tokens',256))
    metadata.update(timing)
    if not checkpoint:metadata['peak_vram_gb']=timing['inference_peak_vram_gb']
    del model,processor
    gc.collect()
    torch.cuda.empty_cache()
    if limit:
        # An abbreviated debugging run is deliberately NOT written to final evidence.
        return dict(predictions=str(path),timing=timing)
    return score_predictions(path,name,split,config.get('coord','norm1000'),metadata,data_root)


def choose_ablation_base(validation):
    if set(validation) != {'E1','E2','E3'}:
        raise ValueError('Require all three validation results before selecting E4 base')
    for result in validation.values():
        if result['split'] != 'val' or result['missing_predictions']:
            raise ValueError('Ablation base must use complete validation evidence')
    return max(validation,key=lambda n:(validation[n]['micro']['f1'],
                                        -validation[n]['meta']['trainable_params']))


def rank_ablation(base):
    if 'lora' not in base:
        raise ValueError('Rank ablation requires a trained base setup')
    variants=[]
    for rank in (4,64):
        cfg=copy.deepcopy(base)
        cfg['name']=f'E4_rank_{rank}'
        cfg['lora']['r']=rank
        cfg['lora']['alpha']=2*rank
        variants.append(cfg)
    return variants


def run_all(data_root='data',checkpoint_root=None):
    meta=json.loads((Path(data_root)/'splits/meta.json').read_text())
    validation={}
    checkpoints={}
    required=[Path('report/evidence')/(n+'.json') for n in ('e0','e1','e2','e3','e4_rank_4','e4_rank_64')]
    if any(p.exists() for p in required):
        raise FileExistsError('Some final evidence already exists; use explicit train/evaluate commands to continue')
    evaluate_checkpoint(config_for('E0'),data_root=data_root)
    for name in ('E1','E2','E3'):
        cfg=config_for(name)
        checkpoints[name]=train(cfg,data_root=data_root,checkpoint_root=checkpoint_root)
        validation[name]=evaluate_checkpoint(cfg,checkpoints[name],split='val',data_root=data_root)
        evaluate_checkpoint(cfg,checkpoints[name],data_root=data_root)
    base_name=choose_ablation_base(validation)
    choice=dict(base=base_name,selected_on='val',dataset_version=meta['dataset_version'],
                variable='rank',values=[4,64],validation_f1={k:v['micro']['f1'] for k,v in validation.items()})
    p=Path('report/diagnostics/ablation_choice.json');p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(choice,indent=2))
    for cfg in rank_ablation(config_for(base_name)):
        checkpoint=train(cfg,data_root=data_root,checkpoint_root=checkpoint_root)
        evaluate_checkpoint(cfg,checkpoint,data_root=data_root)
    from student_vlm.reporting import update_report, prediction_figures
    update_report(data_root=data_root)
    prediction_figures(data_root=data_root)
    subprocess.run([sys.executable,'scripts/check_submission.py'],check=True)
