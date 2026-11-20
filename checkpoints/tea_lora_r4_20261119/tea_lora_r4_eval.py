"""同一冻结模拟器、同一确定性控制器，分别保存裸模型和接控后的闭环历史。"""
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import runpy
import sys
import torch
from peft import PeftModel

def read(p):return json.loads(p.read_text())
def save(p,data):
    p.parent.mkdir(exist_ok=True,parents=True)
    tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    tmp.replace(p)
def weights(base,folder):
    t=read(folder/'training.json')
    assert sha256((folder/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==t['adapter_sha256']
    assert not any('lora_' in n for n,_ in base.named_parameters())
    return PeftModel.from_pretrained(base,folder/'adapter',is_trainable=False).eval()

def core(base,tok,root,run):
    ev=runpy.run_path(str(root/'checkpoints/tea_lora_r2_20261119/tea_lora_r2_eval.py'))
    l=runpy.run_path(str(root/'checkpoints/tea_lora_r1_20261119/tea_lora_round1.py'))
    archive=read(root/'eval/qwen35_4b_tea_baseline_20261119_091933_233315.json')
    out=run/'comparison/第四轮';(out/'eval').mkdir(parents=True,exist_ok=True)
    runpy.run_path(str(run/'tea_lora_r4_calibrate.py'))['calibrate'](base,tok,root,run)
    m=weights(base,run)
    try:
        old=root/'data/tea_chat_sft_v0_2';raw=(old/'validation.jsonl').read_bytes()
        assert sha256(raw).hexdigest()=='fa3294dfcb89b36e2a14693debc6d06b7e78c49bfe933527792c4c68d6a86791'
        topic={x['line']:x['topic'][0] for x in map(json.loads,(old/'provenance.jsonl').read_text().splitlines()) if x['split']=='validation'}
        rows=[]
        for i,s in enumerate(raw.splitlines(),1):
            x=l['encode'](tok,json.loads(s)['messages']);x['domain']=topic[i];rows.append(x)
        olddev=l['score'](m,l['datasets'](root,tok)[1],tok.pad_token_id)
        train=read(run/'training.json')
        selected=read(run/'dev_calibration.json')['selected']
        assert math.isclose(selected['old20'],olddev['nll'],abs_tol=.002)
        score=l['score'](m,rows,tok.pad_token_id)
        assert len(rows)==40 and score['target_tokens']==1112
        save(out/'损失.json',dict(holdout_40=score,dev20=olddev))
        print('R4_HOLDOUT',score,flush=True)
        ev['load_sim'](root)
        env=dict(torch=torch,tokenizer=tok,model=m,ROOT=out,MODEL_ID=l['MODEL_NAME'],REVISION=l['REV'],hardware=archive['hardware'],versions=archive['versions'])
        s=runpy.run_path(str(root/'eval/tea_sim_eval_v1.py'),init_globals=env,run_name='__main__')['tea_sim_report']
        save(out/'原设备模拟指标.json',s['summary'])
        sessions={}
        for key,qs in archive['sessions'].items():
            h=[dict(role='system',content=archive['system_prompt'])];sessions[key]=[]
            for row in qs:
                h.append(dict(role='user',content=row['user']));torch.manual_seed(row['seed'])
                answer,n=ev['generate'](m,tok,h,512,archive['sampling'])
                sessions[key].append(dict(user=row['user'],assistant=answer,new_tokens=n,seed=row['seed']))
                h.append(dict(role='assistant',content=answer))
        save(out/'chat.json',sessions)
        # 复用第三轮固定六题，先前模型结果已归档。
        script=root/'checkpoints/tea_lora_r3b_20261119/tea_lora_r3_transfer.py'
        extras=[]
        for key,f,a,u,rub in runpy.run_path(str(script))['CASES']:
            h=[dict(role=k,content=v) for k,v in [('system',archive['system_prompt']),('user',f),('assistant',a),('user',u)]]
            reply,n=ev['generate'](m,tok,h,192);extras.append(dict(id=key,reply=reply,rubric=rub,user=u,tokens=n))
        save(out/'跨场景六题.json',extras)
    finally:
        m.unload();base.eval()
    ev['test'](base,tok,root,run,'第四轮')
    print('R4_RAW_COMPLETE',score['nll'],s['summary'],flush=True)

def guarded(base,tok,root,run,*,revised=False):
    ev=runpy.run_path(str(root/'checkpoints/tea_lora_r2_20261119/tea_lora_r2_eval.py'))
    s=ev['load_sim'](root)
    b=root/'checkpoints/tea_lora_r3b_20261119'
    path=(run/'controller_v4' if revised else b/'controller')/'tea_controller.py'
    checksum=run/('controller_v4.sha256' if revised else 'controller.sha256')
    assert sha256(path.read_bytes()).hexdigest()==checksum.read_text().strip()
    spec=importlib.util.spec_from_file_location('tea_r4_frozen_controller',path)
    c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
    ok=c.Sensors(True,True,True,True)
    def gen(m,p,payload,limit):return ev['generate'](m,tok,ev['payload_text'](p,payload),limit)[0]
    names=[('第四轮',run),('第二轮',root/'checkpoints/tea_lora_r2_20261119'),('第三轮A',root/'checkpoints/tea_lora_r3_20261119'),('第三轮B',b)]
    for label,folder in names:
        file=run/'comparison'/label/('新控制器闭环.json' if revised else '接控闭环.json')
        if file.is_file() and len(read(file)['rows'])==18:continue
        m=weights(base,folder)
        try:
            out=[]
            for scene,initial,turns in s['SCENES']:
                ctl=c.TeaController()
                state=initial.copy();hist=[];sensors=ok;recipe=None;strength=None
                if state['阶段'] in s['ACTIVE']:
                    recipe=state['当前茶种'];strength=state['当前浓度']
                    r=ctl.turn(f'请泡一杯{recipe}，{strength}',s['decision']('start',recipe,strength),ok)
                    assert r.code=='started'
                    if state['阶段']=='出水中':ctl.controller_event(r.job_id,'dispensing',ok)
                for i,t in enumerate(turns,1):
                    before=ctl.update_sensors(sensors)
                    raw=gen(m,s['PLAN_PROMPT'],dict(历史=hist,用户原话=t['user'],设备阶段=before.phase),96)
                    p=s['parse_plan'](raw);e=t['event']
                    sensors=c.Sensors(False if e=='取走杯子' else sensors.cup,False if e=='水量不足' else sensors.water,True,True)
                    if e:ctl.update_sensors(sensors)
                    rc=ctl.turn(t['user'],raw,sensors)
                    if rc.code=='started':recipe,strength=p['tea'],p['strength']
                    if not rc.job_id:recipe=strength=None
                    if rc.text is not None: spoken=rc.text
                    else:
                        receipt=dict(结果='未请求动作',状态=dict(阶段=rc.phase,杯子在位=sensors.cup,水量足够=sensors.water,当前茶种=recipe,当前浓度=strength),传感器事件=e,联锁=None)
                        spoken=gen(m,s['REPLY_PROMPT'],dict(历史=hist,用户原话=t['user'],解析结果=p,回执=receipt),128)
                    s['sensor_event'](state,e);expected_code=s['simulate'](t['expected'],state)
                    wanted=state['阶段']
                    if e and not rc.job_id and rc.code in ('no_cup','low_water'): 
                        phase_ok=wanted in ('空闲','已停止') and rc.phase=='已拦截'
                    else: phase_ok=rc.phase==wanted
                    state_ok=phase_ok and recipe==state['当前茶种'] and strength==state['当前浓度']
                    allowed={expected_code}
                    if expected_code=='none':
                        allowed={'choose_tea','unsupported','no_substitution'} if t['expected']['action']=='clarify' else {'chat','menu'}
                    elif expected_code=='status' and e:allowed={'no_cup' if e=='取走杯子' else 'low_water'}
                    code_ok=rc.code in allowed
                    out.append(dict(scene=scene,turn=i,user=t['user'],history=list(hist),raw_plan=raw,expected=t['expected'],expected_code=expected_code,code=rc.code,phase=rc.phase,recipe=recipe,strength=strength,expected_state=state.copy(),spoken=spoken,passed=code_ok and state_ok,state_ok=state_ok,code_ok=code_ok))
                    hist.append(dict(用户=t['user'],助手=spoken))
            save(file,dict(passed=sum(v['passed'] for v in out),rows=out,controller_sha256=sha256(path.read_bytes()).hexdigest()))
            print('R4_GUARDED',revised,label,sum(v['passed'] for v in out),len(out),flush=True)
        finally:
            m.unload();base.eval()
