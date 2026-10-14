"""QLoRA setup, module checks and complete projector checkpoint round-trips."""
import json
import re
from pathlib import Path

LLM_TARGET = r'model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)'
VIT_TARGET = r'model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))'


def matched_modules(model, targets='llm'):
    if targets not in ('llm','llm+vit'):
        raise ValueError(f'Unsupported LoRA targets: {targets}')
    regex = LLM_TARGET if targets == 'llm' else f'(?:{LLM_TARGET}|{VIT_TARGET})'
    return [name for name,_ in model.named_modules() if re.fullmatch(regex,name)]


def is_projector(name):
    return 'visual.merger.' in name or 'visual.deepstack_merger_list.' in name


def block_counts(model):
    counts = dict(llm=0, vit=0, projector=0, other=0, total=0)
    for name,p in model.named_parameters():
        key = ('projector' if is_projector(name) else 'vit' if '.visual.' in name
               else 'llm' if '.language_model.' in name or name.startswith('lm_head.') else 'other')
        counts[key] += p.numel()
        counts['total'] += p.numel()
    return counts


def load_base(model_id='Qwen/Qwen3-VL-4B-Instruct'):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError('E0–E4 require a CUDA GPU (Colab T4); CPU checks cannot produce experiment evidence')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                              llm_int8_skip_modules=['visual','lm_head'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16,
        device_map={'':0}, attn_implementation='sdpa')
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'left'
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model,processor


def apply_setup(model, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    targets = config['lora']['targets']
    matches = matched_modules(model,targets)
    expected = 348 if targets == 'llm+vit' else 252
    if len(matches) != expected:
        raise ValueError(f'Expected {expected} LoRA targets, found {len(matches)}: {matches[:8]}')
    for name,module in model.named_modules():
        if '.visual.' in name and getattr(module,'weight',None) is not None:
            if type(module).__name__ == 'Linear4bit':
                raise ValueError(f'ViT/projector must remain unquantized: {name}')
    counts = block_counts(model)
    model.enable_input_require_grads()
    lora = config['lora']
    model = get_peft_model(model,LoraConfig(
        task_type=TaskType.CAUSAL_LM,r=int(lora['r']),lora_alpha=int(lora['alpha']),
        lora_dropout=float(lora['dropout']),target_modules=matches,bias='none'))
    if config['train_projector']:
        for name,p in model.named_parameters():
            if is_projector(name):
                p.data = p.data.to(torch.float32)
                p.requires_grad_(True)
    # PEFT keeps LoRA matrices fp32 for stable fp16 GradScaler updates.
    for name,p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache = False
    return model,dict(block_counts=counts,matched_modules=matches,
                      trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_checkpoint(model, processor, path, run, optimizer=None, scaler=None, scheduler=None):
    import torch
    path = Path(path)
    path.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(path/'adapter')
    processor.save_pretrained(path/'adapter')
    extra = {name:p.detach().float().cpu() for name,p in model.named_parameters()
             if p.requires_grad and is_projector(name)}
    if extra:
        torch.save(extra,path/'projector.pt')
    (path/'run.json').write_text(json.dumps(run,indent=2))
    if optimizer is not None:
        torch.save(dict(optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        scheduler=scheduler.state_dict(),step=run['step'],
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),
                   path/'training_state.pt')


def restore_projector(model, path, required=False):
    import torch
    path = Path(path)/'projector.pt'
    if not path.exists():
        if required:
            raise FileNotFoundError(f'Projector weights missing: {path}')
        return
    extra = torch.load(path,map_location='cpu',weights_only=True)
    named = dict(model.named_parameters())
    expected = {n for n in named if is_projector(n)}
    if set(extra) != expected:
        raise ValueError('Projector checkpoint keys do not match the loaded model')
    with torch.no_grad():
        for name,weight in extra.items():
            if weight.shape != named[name].shape:
                raise ValueError(f'Projector shape mismatch: {name}')
            if required:
                named[name].data = named[name].data.float()
            named[name].copy_(weight.to(device=named[name].device,dtype=named[name].dtype))


def load_finetuned(checkpoint):
    from peft import PeftModel
    checkpoint = Path(checkpoint)
    run = json.loads((checkpoint/'run.json').read_text())
    model,processor = load_base(run['config']['model_id'])
    model = PeftModel.from_pretrained(model,checkpoint/'adapter',is_trainable=False)
    restore_projector(model,checkpoint,required=run['config']['train_projector'])
    return model,processor,run
