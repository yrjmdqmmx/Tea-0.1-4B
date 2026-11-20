"""泡茶机指令基线。先运行茶对话笔记本的 Cell 1–3，再用 %run -i 运行本文件。"""
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo


RECIPES = ("凤凰单丛", "铁观音")
STRENGTHS = ("淡", "标准")
ACTIVE = ("加热中", "出水中")
PLAN_PROMPT = """你是泡茶机的中文意图解析器。结合历史、当前用户原话和设备阶段，只输出一个 JSON 对象。
三个必填字段为 action、tea、strength。action 只可为 chat、clarify、start、stop、status。
chat 表示闲聊、询问知识、假设或转述。clarify 表示用户想泡茶但指代或规格需要澄清。
start 仅用于本轮明确授权立即泡茶，茶种可从明确的上文继承。茶种只支持凤凰单丛和铁观音；
浓度只支持淡和标准，用户未指定时默认标准。没有默认茶种；所选茶种不支持时用 clarify 询问是否换茶。已在执行时也如实解析新请求，交由控制器处理。
stop 表示取消或叫停正在执行的任务；status 表示查询设备状况或进度。辨别否定和改口，以最终意愿为准。
只有 start 填写茶种和浓度，其余动作的 tea 和 strength 都填 null。不要写解释或 Markdown。"""
REPLY_PROMPT = """你是中文智能泡茶机，现在对用户自然地说一两句。输入包含对话历史、用户原话、解析结果和可信的模拟设备回执。
好好回应原话，只有回执确认的动作和状态才可以说已经发生。被拒绝时要说清原因。
需要澄清时只问缺什么。杯子放回或水补足后，等用户重新发令。不要编造传感器数字、倒计时或身体经历。"""
LABELS = {"none": "未请求动作", "started": "已接受启动", "stopped": "已接受停止",
          "already_idle": "当前没有任务", "status": "已读取状态", "no_cup": "未检测到杯子",
          "low_water": "水量不足", "busy": "已有冲泡任务", "invalid": "解析格式无效"}


def decision(action, tea=None, strength=None):
    return {"action": action, "tea": tea, "strength": strength}


def device(phase="空闲", cup=True, water=True):
    return {"阶段": phase, "杯子在位": cup, "水量足够": water,
            "当前茶种": RECIPES[0] if phase in ACTIVE else None,
            "当前浓度": "淡" if phase in ACTIVE else None}


def stop_device(state):
    state.update(阶段="已停止", 当前茶种=None, 当前浓度=None)


def parse_plan(raw):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("重复字段")
        return dict(pairs)
    try:
        d = json.loads(raw, object_pairs_hook=unique)
    except ValueError:
        return None
    if not isinstance(d, dict) or set(d) != {"action", "tea", "strength"}:
        return None
    if d["action"] == "start" and d["tea"] in RECIPES and d["strength"] in STRENGTHS:
        return d
    if d["action"] in ("chat", "clarify", "stop", "status") and d["tea"] is None and d["strength"] is None:
        return d
    return None


def simulate(plan, state):
    if plan is None:
        return "invalid"
    a = plan["action"]
    if a == "status":
        return "status"
    if a == "stop":
        if state["阶段"] not in ACTIVE:
            return "already_idle"
        stop_device(state)
        return "stopped"
    if a != "start":
        return "none"
    if not state["杯子在位"]:
        return "no_cup"
    if not state["水量足够"]:
        return "low_water"
    if state["阶段"] in ACTIVE:
        return "busy"
    state.update(阶段="加热中", 当前茶种=plan["tea"], 当前浓度=plan["strength"])
    return "started"


def turn(user, plan, receipt, phase, event=None):
    return dict(user=user, expected=plan, receipt=receipt, phase=phase, event=event)


CHAT, ASK, STOP, STATUS = (decision(a) for a in ("chat", "clarify", "stop", "status"))
LIGHT = decision("start", "凤凰单丛", "淡")
IRON = decision("start", "铁观音", "标准")
SCENES = [
    ("边聊边泡", device(), [
        turn("凤凰单丛听起来不错。", CHAT, "none", "空闲"),
        turn("就这个，给我泡杯淡的，现在开始吧。", LIGHT, "started", "加热中"),
        turn("等茶的时候陪我聊两句，单丛这个名字挺有意思的。", CHAT, "none", "加热中"),
        turn("我的茶现在到哪一步了？", STATUS, "status", "加热中"),
        turn("等等，先别出水，停一下。", STOP, "stopped", "已停止"),
        turn("现在还在泡吗？", STATUS, "status", "已停止")]),
    ("转述与假设", device(), [
        turn("同事老说‘给我泡杯铁观音’，我只是在学他说话。", CHAT, "none", "空闲"),
        turn("假如我一会儿想喝淡点的凤凰单丛，你能泡吗？先不用做。", CHAT, "none", "空闲")]),
    ("缺茶种再补充", device(), [
        turn("给我泡杯淡的。", ASK, "none", "空闲"),
        turn("凤凰单丛，现在泡吧。", LIGHT, "started", "加热中")]),
    ("否定后改口", device(), [
        turn("来杯铁观音……算了，先不泡了，就聊会儿。", CHAT, "none", "空闲")]),
    ("默认浓度与重复启动", device(), [
        turn("现在给我泡杯铁观音。", IRON, "started", "加热中"),
        turn("再开一杯淡的凤凰单丛，马上开始。", LIGHT, "busy", "加热中")]),
    ("启动时取走杯子", device(), [
        turn("给我泡杯铁观音，现在开始。", IRON, "no_cup", "空闲", "取走杯子")]),
    ("启动时缺水", device(), [
        turn("给我泡杯淡的凤凰单丛，现在开始。", LIGHT, "low_water", "空闲", "水量不足")]),
    ("出水中独立叫停", device("出水中"), [
        turn("快停，别再往杯里倒了！", STOP, "stopped", "已停止")]),
    ("出水中传感器联锁", device("出水中"), [
        turn("现在是什么状态？", STATUS, "status", "已停止", "取走杯子")]),
    ("支持范围之外", device(), [
        turn("现在帮我泡杯龙井。", ASK, "none", "空闲")]),
]


def sensor_event(state, event):
    if event == "取走杯子":
        state["杯子在位"] = False
    elif event == "水量不足":
        state["水量足够"] = False
    if state["阶段"] in ACTIVE and not (state["杯子在位"] and state["水量足够"]):
        stop_device(state)
        return "传感器联锁已停机"
    return None


def verify_fixtures():
    for name, initial, turns in SCENES:
        state = initial.copy()
        for t in turns:
            assert parse_plan(json.dumps(t["expected"])) == t["expected"]
            sensor_event(state, t["event"])
            assert simulate(t["expected"], state) == t["receipt"], (name, t["user"])
            assert state["阶段"] == t["phase"]
    assert parse_plan('{"action":"start","tea":"铁观音","strength":"标准","extra":1}') is None


def generate_text(system, payload, max_tokens):
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    begin = perf_counter()
    with torch.inference_mode():
        ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True,
                             disable_compile=True, pad_token_id=tokenizer.pad_token_id)
    new_ids = ids[0, inputs["input_ids"].shape[1]:].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), round(perf_counter()-begin, 3)


def run_sim():
    verify_fixtures()
    now = lambda: datetime.now(ZoneInfo("Asia/Shanghai"))
    started = now()
    rows = []
    for name, initial, turns in SCENES:
        state, expected_state, history = initial.copy(), initial.copy(), []
        for i, t in enumerate(turns, 1):
            before = state.copy()
            request = {"历史": history, "用户原话": t["user"], "设备阶段": before["阶段"]}
            raw, latency = generate_text(PLAN_PROMPT, request, 96)
            plan = parse_plan(raw)
            interlock = sensor_event(state, t["event"])  # 在模型提交请求之后注入，回执以执行时为准
            seen_by_controller = state.copy()
            code = simulate(plan, state)
            receipt = {"结果": LABELS[code], "状态": state.copy(), "传感器事件": t["event"], "联锁": interlock}
            reply, reply_latency = generate_text(REPLY_PROMPT, {
                "历史": history, "用户原话": t["user"], "解析结果": plan, "回执": receipt}, 128)
            sensor_event(expected_state, t["event"])
            simulate(t["expected"], expected_state)
            checks = {"intent": plan == t["expected"], "receipt": code == t["receipt"],
                      "state": state == expected_state and state["阶段"] == t["phase"]}
            row = dict(scene=name, turn=i, user=t["user"], history=deepcopy(history), before=before,
                       event=t["event"], controller_input=seen_by_controller, raw_plan=raw, plan=plan,
                       expected=t["expected"], expected_receipt=t["receipt"], expected_phase=t["phase"],
                       code=code, receipt=receipt, reply=reply, checks=checks, passed=all(checks.values()),
                       plan_s=latency, reply_s=reply_latency)
            rows.append(row)
            history.append({"用户": t["user"], "助手": reply})
            verdict = "通过" if row["passed"] else "未通过"
            print(f'{len(rows):02} {name}/{i} | {verdict} | {raw} | {LABELS[code]} | {state["阶段"]}\n  {reply}', flush=True)
    meta = dict(model=MODEL_ID, revision=REVISION, hardware=hardware, dtype=str(model.dtype), versions=versions,
                started_at=started.isoformat(), completed_at=now().isoformat(), do_sample=False,
                enable_thinking=False, token_limits={"plan":96, "reply":128}, plan_prompt=PLAN_PROMPT,
                reply_prompt=REPLY_PROMPT, generation_defaults=model.generation_config.to_dict(),
                simulator="离散状态、模拟传感器、无实体设备", code_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    summary = {k: sum(r["checks"][k] for r in rows) for k in ("intent", "receipt", "state")}
    summary.update(total=len(rows), passed=sum(r["passed"] for r in rows),
                   false_start_requests=sum(r["plan"] is not None and r["plan"]["action"]=="start"
                       and r["expected"]["action"]!="start" for r in rows))
    report = dict(meta=meta, summary=summary, cases=SCENES, rows=rows)
    stem = ROOT / "eval" / ("tea_sim_v1_" + started.strftime("%Y%m%d_%H%M%S_%f"))
    lines = ["# 泡茶机模拟指令基线", "", f"{MODEL_ID} · {started.isoformat()}", "",
             f"指令、回执和状态同时符合预期 {summary['passed']}/{len(rows)}。中文回复另行人工核对。"]
    for r in rows:
        lines += ["", f"## {r['scene']} · {r['turn']}", "", r["user"], "",
                  f"模型原始输出 `{r['raw_plan']}`", f"预期 `{json.dumps(r['expected'], ensure_ascii=False)}`",
                  f"模拟回执 `{json.dumps(r['receipt'], ensure_ascii=False)}`", "",
                  r["reply"], "", f"判定 {r['checks']}"]
    for ext, content in ((".json", json.dumps(report, ensure_ascii=False, indent=2)), (".md", "\n".join(lines))):
        path = stem.with_suffix(ext)
        tmp = path.with_suffix(ext + ".tmp")
        tmp.write_text(content + "\n", encoding="utf-8")
        tmp.replace(path)
        assert path.read_text(encoding="utf-8") == content + "\n"
        print("已保存并回读", path, flush=True)
    print("模拟测试完成", json.dumps(summary, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    tea_sim_report = run_sim()
