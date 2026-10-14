"""Batched deterministic generation, preserving every raw model output for the grader."""
import json
import time
from pathlib import Path
from student_vlm.encoding import open_image, user_message


def predict(model, processor, records, output, *, data_root='data', coord='norm1000',
            batch_size=4, max_new_tokens=256):
    import torch
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError('Inference batch size and token budget must be positive')
    ids = [r['image_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate image IDs in inference split')
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Predictions already exist: {output}; do not silently retest/overwrite')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.partial')
    processor.tokenizer.padding_side='left'
    model.eval()
    model.config.use_cache=True
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    with torch.inference_mode(),temporary.open('w') as f:
        for offset in range(0,len(records),batch_size):
            rows=records[offset:offset+batch_size]
            chats=[[user_message(open_image(r,data_root),coord)] for r in rows]
            inputs=dict(processor.apply_chat_template(chats,tokenize=True,add_generation_prompt=True,
                        return_dict=True,return_tensors='pt',padding=True))
            inputs.pop('token_type_ids',None)
            inputs={k:v.to(model.device) for k,v in inputs.items()}
            prompt_len=inputs['input_ids'].shape[1]
            with torch.autocast('cuda',dtype=torch.float16):
                generated=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
            raw=processor.batch_decode(generated[:,prompt_len:],skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            if len(raw) != len(rows):
                raise RuntimeError('Generation returned the wrong number of answers')
            for r,text in zip(rows,raw):
                # Do not fix fences, hallucinated boxes or invalid JSON before scoring.
                f.write(json.dumps({'image_id':r['image_id'],'raw':text},ensure_ascii=False)+'\n')
            print(f'predict {min(offset+len(rows),len(records))}/{len(records)}',flush=True)
    temporary.replace(output)
    return dict(inference_minutes=(time.perf_counter()-start)/60,
                inference_peak_vram_gb=torch.cuda.max_memory_allocated()/1024**3)
