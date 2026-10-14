import copy
import json
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
from peft import PeftModel
from student_vlm.encoding import collate_examples,encode_example
from student_vlm.experiments import config_for,rank_ablation,choose_ablation_base
from student_vlm.modeling import apply_setup,block_counts,matched_modules,restore_projector,save_checkpoint,is_projector
from student_vlm.training import lr_multiplier

CONFIG=json.loads((Path(__file__).parents[1]/'report/diagnostics/model_config.json').read_text())


def meta_model():
    with torch.device('meta'):
        return Qwen3VLForConditionalGeneration(Qwen3VLConfig(**CONFIG))


def tiny_model():
    c=copy.deepcopy(CONFIG)
    c['text_config'].update(hidden_size=32,intermediate_size=64,num_attention_heads=4,
                           num_key_value_heads=2,head_dim=8)
    c['text_config']['rope_scaling']['mrope_section']=[2,1,1]
    c['vision_config'].update(hidden_size=16,intermediate_size=32,num_heads=4,out_hidden_size=32)
    return Qwen3VLForConditionalGeneration(Qwen3VLConfig(**c))


def test_real_meta_architecture_counts_and_regex_targets():
    m=meta_model()
    assert block_counts(m)==dict(llm=4022468096,vit=306242560,projector=109105152,other=0,total=4437815808)
    assert len(matched_modules(m))==252
    assert len(matched_modules(m,'llm+vit'))==348


@pytest.mark.parametrize('name,count', [('E1',33030144),('E2',142135296),('E3',148426752)])
def test_setup_trainability_on_the_real_architecture(name,count):
    m,info=apply_setup(meta_model(),config_for(name))
    assert info['trainable_params']==count
    for n,p in m.named_parameters():
        if is_projector(n):assert p.requires_grad==(name!='E1')
        elif p.requires_grad:assert 'lora_' in n
        if p.requires_grad:assert p.dtype==torch.float32


def test_collator_masks_padding_and_concatenates_images():
    def row(ids,patches):
        return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                    labels=torch.tensor([[-100]+ids[1:]]),mm_token_type_ids=torch.tensor([[1]+[0]*(len(ids)-1)]),
                    pixel_values=torch.ones(patches,8),image_grid_thw=torch.tensor([[1,2,patches//2]]))
    b=collate_examples([row([9,10],4),row([9,11,12],6)],0)
    assert b['input_ids'].tolist()==[[9,10,0],[9,11,12]]
    assert b['labels'].tolist()==[[-100,10,-100],[-100,11,12]]
    assert b['attention_mask'].tolist()==[[1,1,0],[1,1,1]]
    assert b['mm_token_type_ids'].tolist()==[[1,0,0],[1,0,0]]
    assert b['pixel_values'].shape==(10,8) and b['image_grid_thw'].shape==(2,3)


class Processor:
    tokenizer=type('Tokenizer',(),{'convert_tokens_to_ids':lambda self,s:99})()
    def __init__(self,bad=False):self.bad=bad
    def apply_chat_template(self,messages,add_generation_prompt,**kwargs):
        ids=[1,99,99,2] if add_generation_prompt else [1,99,99,2,7,8]
        if self.bad and not add_generation_prompt:ids[0]=5
        return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones(1,len(ids),dtype=torch.long),
                    pixel_values=torch.ones(8,3),image_grid_thw=torch.tensor([[1,2,4]]))


def test_answer_only_mask_and_prefix_failure():
    image=type('Image',(),{'width':640,'height':360})()
    result=encode_example(Processor(),image,[],expected_vision_tokens=2)
    assert result['labels'].tolist()==[[-100,-100,-100,-100,7,8]]
    with pytest.raises(ValueError,match='not a prefix'):
        encode_example(Processor(True),image,[],expected_vision_tokens=2)
    with pytest.raises(ValueError,match='vision tokens'):
        encode_example(Processor(),image,[],expected_vision_tokens=220)


def test_ablation_changes_only_rank_and_derived_alpha():
    base=config_for('E2')
    for c in rank_ablation(base):
        assert c['lora']['alpha']==2*c['lora']['r']
        restored=copy.deepcopy(c)
        restored['name']=base['name'];restored['lora']['r']=base['lora']['r'];restored['lora']['alpha']=base['lora']['alpha']
        assert restored==base
    rows={n:dict(split='val',missing_predictions=0,micro={'f1':score},meta={'trainable_params':cost})
          for n,score,cost in [('E1',.5,1),('E2',.6,2),('E3',.6,3)]}
    assert choose_ablation_base(rows)=='E2'
    rows['E3']['split']='test'
    with pytest.raises(ValueError):choose_ablation_base(rows)


def test_warmup_and_cosine_reaches_zero():
    assert lr_multiplier(0,300,15)==pytest.approx(1/15)
    assert lr_multiplier(14,300,15)==1
    assert lr_multiplier(15,300,15)==1
    assert lr_multiplier(300,300,15)==0


def test_projector_is_saved_and_restored_separately(tmp_path):
    m,_=apply_setup(tiny_model(),config_for('E2'))
    with torch.no_grad():
        for n,p in m.named_parameters():
            if is_projector(n):p.fill_(.1234567)
    processor=type('Processor',(),{'save_pretrained':lambda self,p:None})()
    save_checkpoint(m,processor,tmp_path,{'config':config_for('E2')})
    reloaded=PeftModel.from_pretrained(tiny_model().half(),tmp_path/'adapter')
    restore_projector(reloaded,tmp_path,required=True)
    for n,p in reloaded.named_parameters():
        if is_projector(n):
            assert p.dtype==torch.float32
            assert torch.equal(p,torch.full_like(p,.1234567))
    (tmp_path/'projector.pt').unlink()
    with pytest.raises(FileNotFoundError):restore_projector(reloaded,tmp_path,required=True)


def test_real_tiny_qwen_backward_has_llm_vit_and_projector_gradients():
    torch.set_num_threads(1)
    m,_=apply_setup(tiny_model(),config_for('E3'))
    m.train()
    ids=torch.tensor([[151652,151655,151655,151655,151655,151653,10,11,12]])
    labels=ids.clone();labels[:,:7]=-100
    out=m(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels,
          pixel_values=torch.randn(16,1536),image_grid_thw=torch.tensor([[1,4,4]]),
          mm_token_type_ids=(ids==151655).long())
    assert torch.isfinite(out.loss)
    out.loss.backward()
    nonzero=[n for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().sum()>0]
    assert any('.language_model.' in n and 'lora_' in n for n in nonzero)
    assert any('.visual.blocks.' in n and 'lora_' in n for n in nonzero)
    assert any(is_projector(n) for n in nonzero)
    assert all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)


def test_completed_resume_returns_checkpoint_without_retraining(tmp_path):
    from student_vlm.training import train
    config=config_for('E2')
    data=tmp_path/'data/splits';data.mkdir(parents=True)
    (data/'train.jsonl').write_text('{"image_id":"not-used"}\n')
    (data/'meta.json').write_text('{"dataset_version":"unit-checkpoint-only"}')
    checkpoint=tmp_path/'finished';checkpoint.mkdir()
    (checkpoint/'adapter').mkdir()
    (checkpoint/'projector.pt').touch()
    (checkpoint/'run.json').write_text(json.dumps(dict(config=config,dataset_version='unit-checkpoint-only',step=300)))
    assert train(config,data_root=tmp_path/'data',resume=checkpoint)==checkpoint
    (checkpoint/'projector.pt').unlink()
    with pytest.raises(FileNotFoundError):train(config,data_root=tmp_path/'data',resume=checkpoint)
