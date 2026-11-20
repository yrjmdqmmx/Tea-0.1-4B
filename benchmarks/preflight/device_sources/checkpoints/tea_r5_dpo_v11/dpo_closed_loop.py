"""在现有 Colab L4 上用两个冻结适配器独立生成所有提议与闲聊。"""
from pathlib import Path
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util, json, runpy, sys


def load_control(source):
    name="tea_voice_controller_live"
    spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def closed(model,tok,root,folder,c,helpers,sim):
    healthy=c.Sensors(True,True,True,True)
    pairs=runpy.run_path(str(root/"checkpoints/tea_lora_r5_final_20261119/tea_lora_r5_old_probes.py"))["PAIRS"]
    def infer(msgs,limit):
        return helpers["generate"](model,tok,msgs,limit)[0]

    def say(ctl,h,u,sensors,event=None):
        # 故障按冻结场景时序发生在计划推理之后、提议执行之前。
        wire=sensors
        def predict(msgs,lim):
            nonlocal wire
            answer=infer(msgs,lim)
            if lim==96 and event in ("取走杯子","水量不足"):
                wire=replace(wire,**({"cup":False} if event=="取走杯子" else {"water":False}))
            return answer
        audible=[]
        rc,word,trace=ctl.run_turn(u,h,predict,lambda:wire,plan_prompt=sim["PLAN_PROMPT"],
                                  reply_prompt=sim["REPLY_PROMPT"],emit=audible.append)
        assert word=="".join(audible)
        for ms in (trace["plan_input"],trace["reply_input"]):
            if ms:
                assert ms[0]["role"]=="system" and ms[0]["content"].splitlines()[-1].startswith("[实时状态:")
        h.append(dict(用户=u,助手=word))
        return dict(user=u,receipt=asdict(rc),active_job=rc.active_job_id,recipe=ctl._recipe,
                    strength=ctl._strength,spoken=word,**trace),wire

    simulation=[]
    for scene,initial,turns in sim["SCENES"]:
        ctl=c.TeaController();wire=healthy;history=[];gold=initial.copy()
        if gold["阶段"] in sim["ACTIVE"]:
            t,w=gold["当前茶种"],gold["当前浓度"]
            start=ctl.turn(f"请泡一杯{t}，{w}",sim["decision"]("start",t,w),healthy)
            assert start.code=="started"
            if gold["阶段"]=="出水中":ctl.controller_event(start.job_id,"dispensing",healthy)
        for ix,item in enumerate(turns,1):
            obj,wire=say(ctl,history,item["user"],wire,item["event"])
            sim["sensor_event"](gold,item["event"])
            code=sim["simulate"](item["expected"],gold)
            permitted={code}
            if code=="none":
                permitted={"choose_tea","unsupported","no_substitution"} if item["expected"]["action"]=="clarify" else {"chat","menu","capability"}
            if code=="status" and item["event"]:
                permitted={"no_cup" if item["event"]=="取走杯子" else "low_water"}
            phase=obj["receipt"]["phase"]
            phase_ok=phase==gold["阶段"] or (item["event"] and not obj["active_job"] and phase=="已拦截" and gold["阶段"] in ("空闲","已停止"))
            obj.update(scene=scene,index=ix,expected=gold.copy(),expected_code=code,
                       passed=bool(obj["receipt"]["code"] in permitted and phase_ok and obj["recipe"]==gold["当前茶种"] and obj["strength"]==gold["当前浓度"]))
            simulation.append(obj)
    pairs_out=[]
    for key,first,last,want,tea in pairs:
        ctl=c.TeaController();hist=[]
        a,_=say(ctl,hist,first,healthy);b,_=say(ctl,hist,last,healthy)
        allowed={"ask":{"choose_tea"},"unsupported":{"choose_tea","unsupported","no_substitution"},
                 "idle":{"chat","status","selected","choose_tea"},"started":{"started"}}
        ok=not a["active_job"] and b["receipt"]["code"] in allowed[want]
        ok=bool(ok and ((b["active_job"] and b["recipe"]==tea and b["strength"]=="标准") if want=="started" else not b["active_job"]))
        pairs_out.append(dict(id=key,expected=want,expected_tea=tea,passed=ok,turns=[a,b]))
    assert len(simulation)==18 and len(pairs_out)==8
    return dict(simulation=simulation,pairs=pairs_out,controller_sha256=sha256((folder/"tea_controller.py").read_bytes()).hexdigest())
