"""在相同上下文下生成三版专项回答，保留原文、动作与模拟回执。"""
import hashlib
import json
from pathlib import Path
import runpy
import torch
from peft import PeftModel

def generate(model,tok,msgs,limit,sampling=None):
    prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inp=tok(prompt,add_special_tokens=False,return_tensors="pt").to(model.device)
    options=sampling or dict(do_sample=False,max_new_tokens=limit,pad_token_id=tok.pad_token_id,
                             use_cache=True,disable_compile=True)
    with torch.inference_mode():
        ids=model.generate(**inp,**options)[0,inp.input_ids.shape[1]:].tolist()
    return tok.decode(ids,skip_special_tokens=True).strip(),len(ids)

def load_sim(project):
    p=project/"eval/tea_sim_eval_v1.py"
    assert hashlib.sha256(p.read_bytes()).hexdigest()=="3748387842f5d06e73cf18a05753a8e8ba25da7233bb8ab273b3c08c861d520b"
    return runpy.run_path(str(p))

def payload_text(s,p):
    return [dict(role="system",content=s),dict(role="user",content=json.dumps(p,ensure_ascii=False))]

def test(base,tok,project,run,variant):
    sim=load_sim(project)
    data=project/"data/tea_chat_sft_r2"
    cases=json.loads((data/"专项测试.json").read_text())
    out=run/"comparison"/variant
    out.mkdir(parents=True,exist_ok=True)
    file=out/"专项原始回答.json"
    if file.exists():
        assert len(json.loads(file.read_text()))==len(cases)
        return
    assert not any("lora_" in n for n,_ in base.named_parameters())
    model=base
    if variant!="原版":
        parent=project/"checkpoints/tea_lora_r1_20261119" if variant=="首轮" else run
        expected=json.loads((parent/"training.json").read_text())["adapter_sha256"]
        assert hashlib.sha256((parent/"adapter/adapter_model.safetensors").read_bytes()).hexdigest()==expected
        model=PeftModel.from_pretrained(base,parent/"adapter")
    try:
        model.eval()
        rows=[]
        for c in cases:
            key,topic=c[:2]
            if topic in ("接话","茶名"):
                first,prev,user,rubric=c[2:]
                msgs=[dict(role="system",content="说中文，像朋友一样自然聊天。听清对方的意思，回答直接一点，不确定就直说。")]
                if first: msgs.extend([dict(role="user",content=first),dict(role="assistant",content=prev)])
                reply,n=generate(model,tok,msgs+[dict(role="user",content=user)],224)
                row=dict(id=key,topic=topic,user=user,rubric=rubric,reply=reply,tokens=n)
            else:
                hist,user,action,tea,strength,rubric=c[2:]
                hist=[dict(用户=u,助手=a) for u,a in hist]
                state=sim["device"]("加热中" if action=="stop" else "空闲")
                raw,_=generate(model,tok,payload_text(sim["PLAN_PROMPT"],dict(历史=hist,用户原话=user,设备阶段=state["阶段"])),96)
                plan=sim["parse_plan"](raw)
                code=sim["simulate"](plan,state)
                receipt=dict(结果=sim["LABELS"][code],状态=state,传感器事件=None,联锁=None)
                reply,n=generate(model,tok,payload_text(sim["REPLY_PROMPT"],dict(历史=hist,用户原话=user,解析结果=plan,回执=receipt)),128)
                expected=sim["decision"](action,tea,strength)
                row=dict(id=key,topic=topic,user=user,rubric=rubric,reply=reply,tokens=n,plan=plan,raw_plan=raw,
                         expected=expected,plan_ok=plan==expected,receipt=receipt)
            rows.append(row)
            print(variant,key,row.get("plan_ok",""),n,flush=True)
        tmp=file.with_suffix(".part")
        tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        tmp.replace(file)
    finally:
        if model is not base: model.unload()
        base.eval()
