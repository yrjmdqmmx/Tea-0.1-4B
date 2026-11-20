"""第五轮 BF16 LoRA。每个完整 epoch 各样本仅入场一次，三组开发数据选模。"""
from pathlib import Path
from hashlib import sha256
import json, math, random, runpy, shutil
import torch
from peft import LoraConfig,get_peft_model
from transformers import get_cosine_schedule_with_warmup

def train(base,tok,root,run):
    lib=runpy.run_path(str(root/'checkpoints/tea_lora_r1_20261119/tea_lora_round1.py'))
    ev=runpy.run_path(str(root/'checkpoints/tea_lora_r2_20261119/tea_lora_r2_eval.py'))
    sim=ev['load_sim'](root)
    read=lambda p:json.loads(p.read_text())
    lines=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
    data=root/'data/tea_chat_sft_r5'
    meta=read(data/'manifest.json')
    if (run/'training.json').exists():return read(run/'training.json')
    assert not (run/'steps.jsonl').exists(), '已有部分训练记录，请使用独立运行目录'
    assert base.dtype==torch.bfloat16 and torch.cuda.get_device_name()=='NVIDIA L4'
    assert not any('lora_' in n for n,_ in base.named_parameters())
    assert sha256((data/'train_merged.jsonl').read_bytes()).hexdigest()==meta['sha256']
    def encode(m,domain):
        row=lib['encode'](tok,m);row['domain']=domain;return row
    rows=[encode(v['messages'],p['topic']) for v,p in zip(lines(data/'train_merged.jsonl'),lines(data/'provenance.jsonl'),strict=True)]
    assert len(rows)==meta['training']
    unique=len(rows)
    assert meta['per_epoch']==unique and meta['replay'] is False
    _,old=lib['datasets'](root,tok)
    dev=read(root/'data/tea_chat_sft_r4/dev_actions.json')
    system='说中文，像朋友一样自然聊天。听清对方的意思，回答直接一点，不确定就直说。'
    language=[]
    for item in (data/'开发对话.txt').read_text().splitlines():
        q,a=item.split('|')
        language.append(encode([dict(role='system',content=system),dict(role='user',content=q),dict(role='assistant',content=a)],'language'))
    questions=[]
    for user,answer,query,phase,act,tea,w in dev:
        payload=dict(历史=[dict(用户=user,助手=answer)],用户原话=query,设备阶段=phase)
        questions.append((ev['payload_text'](sim['PLAN_PROMPT'],payload),sim['decision'](act,tea,w)))
    extra=[encode(q+[dict(role='assistant',content=json.dumps(p,ensure_ascii=False))],'action') for q,p in questions]
    pad=tok.pad_token_id
    baseline=lib['score'](base,old,pad)['nll']; assert abs(baseline-3.83680324)<.002
    base_action=lib['score'](base,extra,pad)['nll']
    base_language=lib['score'](base,language,pad)['nll']
    seed=20261123;torch.manual_seed(seed)
    wanted={'q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj'}
    targets=[n for n,m in base.named_modules() if 'language_model.layers.' in n and n.split('.')[-1] in wanted and isinstance(m,torch.nn.Linear)]
    assert len(targets)==200
    cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM',revision=lib['REV'])
    model=get_peft_model(base,cfg)
    try:
        assert all(not p.count_nonzero().item() for n,p in model.named_parameters() if 'lora_B.' in n)
        params=[p for n,p in model.named_parameters() if p.requires_grad]
        full=math.ceil(len(rows)/8)
        stops={full,full+math.ceil(full/4),full+math.ceil(full/2)}
        total=max(stops); lr=4.5e-5
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
        schedule=get_cosine_schedule_with_warmup(opt,16,total)
        cfg=dict(base=lib['MODEL_NAME'],revision=lib['REV'],parent_adapter=None,fresh=True,lora_B_verified_zero=True,seed=seed,rows=unique,per_epoch=len(rows),train_sha256=meta['sha256'],rank=16,alpha=32,dropout=.05,lr=lr,optimizer='AdamW',warmup=16,replay=False,schedule='cosine',clip=1.,effective_batch=8,trainable=sum(p.numel() for p in params),steps=total,selection_steps=sorted(stops),criterion='.62*old20/base + .20*language12/base + .06*action12_nll/base + .12*(1-action12 accuracy)',baseline=baseline,base_action=base_action,base_language=base_language,train_sha_note='only assistant responses and EOS',start=lib['now']())
        lib['write_json'](run/'settings.json',cfg)
        def log(**record):
            record['time']=lib['now']()
            with (run/'steps.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            if record['step']%20==0 or record['stage']!='step':print(record,flush=True)
        best=float('inf');checks=[];step=0
        torch.cuda.reset_peak_memory_stats()
        for epoch in (1,2):
            ordered=rows.copy();random.Random(seed+epoch).shuffle(ordered)
            for off in range(0,len(ordered),8):
                model.train();group=sorted(ordered[off:off+8],key=lambda x:len(x['ids']))
                n=sum(x['n'] for x in group);opt.zero_grad(set_to_none=True)
                loss_sum=0.;i=0
                while i<len(group):
                    width=max(len(x['ids']) for x in group[i:i+4])
                    count=4 if width<=130 else (2 if width<=260 else 1)
                    micro=group[i:i+count];nt=sum(x['n'] for x in micro)
                    loss=model(**lib['batch'](micro,pad),use_cache=False).loss
                    (loss*(nt/n)).backward();loss_sum+=loss.item()*nt;i+=len(micro)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True).item()
                used_lr=opt.param_groups[0]['lr'];opt.step();schedule.step();step+=1
                log(stage='step',step=step,nll=loss_sum/n,lr=used_lr,grad_norm=norm)
                if step in stops:
                    val=lib['score'](model,old,pad)['nll'];action=lib['score'](model,extra,pad)['nll']
                    answers=[ev['generate'](model,tok,q,96)[0] for q,p in questions]
                    hits=sum(sim['parse_plan'](s)==p for s,(q,p) in zip(answers,questions))
                    language_nll=lib['score'](model,language,pad)['nll']
                    metric=.62*val/baseline+.20*language_nll/base_language+.06*action/base_action+.12*(1-hits/len(dev))
                    item=dict(step=step,language12=language_nll,old20=val,action_dev_nll=action,action_hits=hits,total=len(dev),criterion=metric,answers=answers)
                    checks.append(item);log(stage='dev',step=step,language12=language_nll,old20=val,hits=hits,criterion=metric)
                    if metric<best:
                        best=metric;best_step=step
                        model.save_pretrained(run/'adapter',safe_serialization=True,save_embedding_layers=False)
                if step==total:break
            if step==total:break
        result=dict(**cfg,end=lib['now'](),best_step=best_step,checks=checks,peak_GB=torch.cuda.max_memory_allocated()/1e9,adapter_sha256=sha256((run/'adapter/adapter_model.safetensors').read_bytes()).hexdigest())
        lib['write_json'](run/'training.json',result)
        shutil.copy2(run/'steps.jsonl',root/'logs'/f'{run.name}.jsonl')
        print('R5_TRAINED',best_step,step,result['adapter_sha256'],flush=True)
        return result
    finally:
        model.unload();base.eval()
