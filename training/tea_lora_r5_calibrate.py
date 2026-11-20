"""只用开发集校准 LoRA 的整体强度，保留原始配置和所有四个候选的证据。"""
import json, runpy
from hashlib import sha256
import torch
from peft.tuners.lora.layer import LoraLayer

def calibrate(base,tok,root,run):
    lib=runpy.run_path(str(root/'checkpoints/tea_lora_r1_20261119/tea_lora_round1.py'))
    ev=runpy.run_path(str(root/'checkpoints/tea_lora_r2_20261119/tea_lora_r2_eval.py'))
    helpers=runpy.run_path(str(root/'checkpoints/tea_lora_r4_20261119/tea_lora_r4_eval.py'))
    read,save=helpers['read'],helpers['save']
    report=run/'dev_calibration.json'
    config_path=run/'adapter/adapter_config.json'
    if report.is_file():
        assert read(config_path)['lora_alpha']==read(report)['selected']['alpha']
        return
    train=read(run/'training.json');base_nll=train['baseline'];base_action=train['base_action']
    sim=ev['load_sim'](root);_,general=lib['datasets'](root,tok)
    dev=read(root/'data/tea_chat_sft_r4/dev_actions.json')
    questions=[];extra=[]
    for f,a,u,phase,act,tea,strength in dev:
        ms=ev['payload_text'](sim['PLAN_PROMPT'],dict(历史=[dict(用户=f,助手=a)],用户原话=u,设备阶段=phase))
        gold=sim['decision'](act,tea,strength)
        row=lib['encode'](tok,ms+[dict(role='assistant',content=json.dumps(gold,ensure_ascii=False))])
        row['domain']='action';extra.append(row);questions.append((ms,gold))
    system='说中文，像朋友一样自然聊天。听清对方的意思，回答直接一点，不确定就直说。'
    language=[]
    for line in (root/'data/tea_chat_sft_r5/开发对话.txt').read_text().splitlines():
        q,a=line.split('|')
        ms=[dict(role=k,content=s) for k,s in [('system',system),('user',q),('assistant',a)]]
        language.append(dict(lib['encode'](tok,ms),domain='language'))
    backup=run/'adapter_config_before_calibration.json'
    if not backup.is_file():save(backup,read(config_path))
    original=read(backup)
    assert original['lora_alpha']==32
    if read(config_path)!=original:save(config_path,original)
    candidates=[]
    m=helpers['weights'](base,run)
    try:
        layers=[x for x in m.modules() if isinstance(x,LoraLayer)]
        assert len(layers)==200 and read(config_path)['lora_alpha']==32
        for alpha in [32,24,28,36]:
            for layer in layers:layer.set_scale('default',alpha/32)
            old=lib['score'](m,general,tok.pad_token_id)['nll']
            action=lib['score'](m,extra,tok.pad_token_id)['nll']
            ans=[ev['generate'](m,tok,ms,96)[0] for ms,_ in questions]
            hit=sum(sim['parse_plan'](x)==p for x,(_,p) in zip(ans,questions))
            lang=lib['score'](m,language,tok.pad_token_id)['nll']
            criterion=.62*old/base_nll+.20*lang/train['base_language']+.06*action/base_action+.12*(1-hit/len(dev))
            candidates.append(dict(alpha=alpha,language12=lang,old20=old,action_dev_nll=action,action_hits=hit,criterion=criterion,answers=ans))
            print('R5_DEV_ALPHA',alpha,old,hit,criterion,flush=True)
    finally:
        m.unload();base.eval()
    selected=min(candidates,key=lambda item:item['criterion'])
    cfg=read(config_path)
    if selected['alpha']!=32:
        cfg['lora_alpha']=selected['alpha'];save(config_path,cfg)
    save(report,dict(candidates=candidates,selected=selected,adapter_config_sha256=sha256(config_path.read_bytes()).hexdigest(),pretrained_rank=16,training_alpha=32,weights_sha256=train['adapter_sha256']))
    print('R5_CALIBRATED',selected['alpha'],flush=True)
