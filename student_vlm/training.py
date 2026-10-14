"""Manual fp16 QLoRA training: answer-only loss, finite gradients and resumable state."""
import copy
import json
import math
import random
import time
from functools import partial
from pathlib import Path
import numpy as np
from scripts.labcommon import read_jsonl
from student_vlm.encoding import RoadDataset, collate_examples
from student_vlm.modeling import apply_setup, load_base, save_checkpoint


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_multiplier(step, max_steps, warmup):
    if step < warmup:
        return (step+1)/max(1,warmup)
    progress = (step-warmup)/max(1,max_steps-warmup)
    return .5*(1+math.cos(math.pi*min(1,progress)))


def train(config, *, data_root='data', output_root='work', resume=None, checkpoint_root=None):
    import torch
    from torch.utils.data import DataLoader
    from student_vlm.modeling import restore_projector
    cfg = copy.deepcopy(config)
    for key in ('batch_size','grad_accum','max_steps'):
        if int(cfg[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if cfg['lr'] <= 0:
        raise ValueError('lr must be positive')
    seed_everything(cfg['seed'])
    records = read_jsonl(Path(data_root)/'splits/train.jsonl')
    fraction = float(cfg.get('train_fraction',1))
    if not 0 < fraction <= 1 or not records:
        raise ValueError('Need a nonempty train split and 0 < train_fraction <= 1')
    if fraction < 1:
        order = np.random.default_rng(cfg['seed']).permutation(len(records))
        records = [records[int(i)] for i in order[:max(1,round(len(records)*fraction))]]
    model,processor = load_base(cfg['model_id'])
    model,setup = apply_setup(model,cfg)
    if resume:
        prior = json.loads((Path(resume)/'run.json').read_text())
        if prior['config'] != cfg:
            raise ValueError('Resume config must exactly match the original training config')
        model.load_adapter(Path(resume)/'adapter',adapter_name='default',is_trainable=True)
        restore_projector(model,resume,required=cfg['train_projector'])
        for name,p in model.named_parameters():
            if cfg['train_projector'] and ('visual.merger.' in name or 'visual.deepstack_merger_list.' in name):
                p.data = p.data.float()
                p.requires_grad_(True)
    ds = RoadDataset(records,processor,data_root,cfg.get('coord','norm1000'))
    def loader_for(epoch):
        generator = torch.Generator().manual_seed(cfg['seed']+epoch)
        return DataLoader(ds,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,
                          generator=generator,collate_fn=partial(collate_examples,
                          pad_token_id=processor.tokenizer.pad_token_id))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=0)
    warmup = max(1,math.ceil(.05*cfg['max_steps']))
    schedule = torch.optim.lr_scheduler.LambdaLR(opt,lambda step:lr_multiplier(step,cfg['max_steps'],warmup))
    scaler = torch.amp.GradScaler('cuda')
    out = Path(output_root)/cfg['name'].lower()
    out.mkdir(parents=True,exist_ok=True)
    step,consumed,elapsed_prior = 0,0,0.0
    if resume:
        state = torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=True)
        opt.load_state_dict(state['optimizer'])
        schedule.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        step,consumed = state['step'],prior['microbatches_consumed']
        elapsed_prior = prior['train_minutes']*60
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    # Deterministic epoch-local shuffling permits resuming without replaying preprocessing of skipped batches.
    batches_per_epoch = math.ceil(len(ds)/cfg['batch_size'])
    epoch,skip = divmod(consumed,batches_per_epoch)
    loader = loader_for(epoch)
    iterator = iter(loader)
    for _ in range(skip):next(iterator)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    dataset_version = json.loads((Path(data_root)/'splits/meta.json').read_text())['dataset_version']
    log_path = out/'loss.jsonl'
    if log_path.exists() and not resume:
        raise FileExistsError(f'{log_path} already exists; use a new output directory or --resume')
    if resume:
        if prior['dataset_version'] != dataset_version:
            raise ValueError('Dataset version changed since the checkpoint')
        if log_path.exists():
            rows = [line for line in log_path.read_text().splitlines()
                    if json.loads(line)['step'] <= step]
            log_path.write_text(''.join(line+'\n' for line in rows))
    run = dict(config=cfg,dataset_version=dataset_version,train_images=len(records),**setup)
    opt.zero_grad(set_to_none=True)
    while step < cfg['max_steps']:
        loss_sum = 0.0
        for _ in range(cfg['grad_accum']):
            try:batch=next(iterator)
            except StopIteration:
                epoch += 1
                iterator=iter(loader_for(epoch))
                batch=next(iterator)
            batch = {k:v.to('cuda') for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss before optimizer step {step+1}')
            loss_sum += float(loss.detach())
            scaler.scale(loss/cfg['grad_accum']).backward()
            consumed += 1
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(params,1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient norm at step {step+1}; lower LR/check AMP')
        scaler.step(opt)
        scaler.update()
        schedule.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        minutes=(elapsed_prior+time.perf_counter()-started)/60
        row=dict(step=step,loss=loss_sum/cfg['grad_accum'],lr=opt.param_groups[0]['lr'],
                 grad_norm=float(norm),minutes=minutes)
        with log_path.open('a') as f:f.write(json.dumps(row)+'\n')
        if step % 10 == 0 or step == 1:print(cfg['name'],row,flush=True)
        run.update(step=step,microbatches_consumed=consumed,train_minutes=minutes,
                   peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3)
        if step % 100 == 0 or step == cfg['max_steps']:
            save_checkpoint(model,processor,out/f'step-{step:04d}',run,opt,scaler,schedule)
            if checkpoint_root:
                save_checkpoint(model,processor,Path(checkpoint_root)/cfg['name'].lower()/'latest',
                                run,opt,scaler,schedule)
    (out/'run.json').write_text(json.dumps(run,indent=2))
    final=out/f"step-{cfg['max_steps']:04d}"
    del model,opt,params,batch,loss
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return final
