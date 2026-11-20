"""冻结 5A 的软件闭环、物理轮证据及 5B 本地四项计分。仅依赖标准库。"""
import argparse
import hashlib
import json
import runpy
import uuid
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from unittest.mock import patch

PREFLIGHT = Path(__file__).resolve().parents[1] / "preflight"
SOURCES = PREFLIGHT / "device_sources"
SUITE = "Tea-Device-44"
ERRORS = ("误播", "漏启动", "误启动", "指错茶")
CLARIFY = ["choose_tea", "unsupported", "no_substitution"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources():
    for name, expected in read(PREFLIGHT / "device_sources.json").items():
        if hashlib.sha256((SOURCES / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"5A source changed: {name}")
    sim = runpy.run_path(str(SOURCES / "eval/tea_sim_eval_v1.py"))
    pairs = runpy.run_path(str(SOURCES / "checkpoints/tea_lora_r5_final_20261119/tea_lora_r5_old_probes.py"))["PAIRS"]
    closed = runpy.run_path(str(SOURCES / "checkpoints/tea_r5_dpo_v11/dpo_closed_loop.py"))
    held = runpy.run_path(str(SOURCES / "logs/step5a/heldwire5.py"))["held_device"]
    suite = read(SOURCES / "data/tea_chat_sft_r5/盲测题.json")
    return sim, pairs, closed, held, suite


def reference(user, codes, phase="空闲", tea=None, strength=None, **extra):
    return dict(user=user, codes=codes, phase=phase, tea=tea, strength=strength,
                should_start=codes == ["started"], **extra)


def cases():
    sim, pairs, _, _, suite = load_sources()
    rows = []
    for scene, initial, steps in sim["SCENES"]:
        state = initial.copy()
        for step in steps:
            sim["sensor_event"](state, step["event"])
            code = sim["simulate"](step["expected"], state)
            codes = [code]
            if code == "none":
                codes = CLARIFY if step["expected"]["action"] == "clarify" else ["chat", "menu", "capability"]
            if code == "status" and step["event"]:
                codes = ["no_cup" if step["event"] == "取走杯子" else "low_water"]
            ref = reference(step["user"], codes, state["阶段"], state["当前茶种"], state["当前浓度"],
                            scene=scene, event=step["event"], initial=initial, action=step["expected"],
                            allow_blocked=bool(step["event"] and state["阶段"] in ("空闲", "已停止")))
            rows.append(dict(id=f"device-S{len(rows)+1:02}", group="simulation", turns=[ref]))
    allowed = {"ask": ["choose_tea"], "unsupported": CLARIFY,
               "idle": ["chat", "status", "selected", "choose_tea"], "started": ["started"]}
    for key, first, last, target, tea in pairs:
        rows.append(dict(id=f"device-{key}-1", group="pairs", turns=[reference(first, None)]))
        start = target == "started"
        rows.append(dict(id=f"device-{key}-2", group="pairs", turns=[
            reference(last, allowed[target], "加热中" if start else "空闲", tea, "标准" if start else None)
        ], first_id=f"device-{key}-1"))
    for key, first, ack, last, action, tea, strength in suite["device"]:
        start = action == "start"
        codes = ["started"] if start else CLARIFY if action == "clarify" else ["selected", "chat", "status"]
        if key == "P10":
            codes = ["capability"]
        rows.append(dict(id=f"device-{key}", group="heldout", fixed_history_reply=ack, turns=[
            reference(first, None),
            reference(last, codes, "加热中" if start else "空闲", tea, strength,
                      must_say="支持淡档" if key == "P10" else None),
        ]))
    if len(rows) != 44 or sum(len(r["turns"]) for r in rows) != 54:
        raise ValueError("5A case inventory changed")
    return rows


def drive(generate, emit):
    """Call generate(call_id, messages, limit) -> evidence, emit(case, turns) after every scoring unit."""
    sim, _, closed, held, suite = load_sources()
    controller_dir = SOURCES / "exports/controller"
    ctl = closed["load_control"](controller_dir / "tea_controller.py")
    refs = cases()
    cursor = iter((case, n, gold) for case in refs for n, gold in enumerate(case["turns"], 1))
    run_turn = ctl.TeaController.run_turn
    pending = []
    def capture(self, user, history, infer, sensors, **kwargs):
        case, n, gold = next(cursor)
        if user != gold["user"]:
            raise ValueError(f"turn order mismatch {case['id']}/{n}")
        calls = []
        def model_callback(messages, limit):
            kind = "plan" if limit == 96 else "draft"
            call_id = f"{case['id']}/{n}/{kind}"
            evidence = generate(call_id, messages, limit)
            calls.append(dict(call_id=call_id, messages=messages, **evidence))
            return evidence["output"]
        # Original closed() wraps its predict callback to apply the sensor event after planning.
        # Dispatch via that wrapper. It calls helper_generate, which reaches model_callback below.
        callback_stack.append(model_callback)
        try:
            rc, spoken, trace = run_turn(self, user, history, infer, sensors, **kwargs)
        finally:
            callback_stack.pop()
        pending.append(dict(user=user, receipt=asdict(rc), active_job=rc.active_job_id,
                            recipe=self._recipe, strength=self._strength, spoken=spoken, calls=calls, **trace))
        if n == len(case["turns"]):
            emit(case, list(pending))
            pending.clear()
        return rc, spoken, trace
    callback_stack = []
    def helper_generate(_model, _tok, messages, limit):
        return callback_stack[-1](messages, limit), 0
    helpers = dict(generate=helper_generate)
    ids = iter(uuid.uuid5(uuid.NAMESPACE_OID, "tea-dpo-v11-ab-"+str(i)) for i in range(100))
    # Fixed virtual time keeps model latency and resume delays out of the 90-second mention TTL.
    with patch.object(ctl.TeaController, "run_turn", capture), patch.object(ctl, "monotonic", return_value=1.0):
        with patch.object(ctl, "uuid4", side_effect=lambda: next(ids)):
            closed["closed"](None, None, SOURCES, controller_dir, ctl, helpers, sim)
        held(None, None, suite, helpers, sim, ctl)
    if pending or next(cursor, None) is not None:
        raise ValueError("missing physical turns")


def turn_score(t, ref):
    if t["user"] != ref["user"]:
        raise ValueError("device turn user mismatch")
    rc = t["receipt"]
    active = t["active_job"]
    started = rc["code"] == "started"
    # Original 5A fixed-turn broadcast predicates applied also to both held-out turns.
    events = {key: [] for key in ERRORS}
    if started != ref["should_start"]:
        events["误启动" if started else "漏启动"].append(rc["code"])
    if started and t["recipe"] != ref["tea"] or active and ref["tea"] and t["recipe"] != ref["tea"]:
        events["指错茶"].append(dict(actual=t["recipe"], expected=ref["tea"]))
    if t["spoken"] != "".join(x for x, _ in groupby(s["spoken"] for s in t["sentences"])):
        raise ValueError("audible transcript differs from per-sentence trace")
    for s in t["sentences"]:
        text, job, state = s["spoken"], s["job_id"], s["phase"]
        bad = False
        for words, expected in [
            (("开始加热", "当前加热中", "已加热", "已经加热好"), "加热中"),
            (("当前出水中", "正在出水", "开始出水"), "出水中"),
            (("泡好了", "泡好啦", "已泡好", "泡茶完成", "请小心取杯", "可以取杯", "可以喝了"), "已完成"),
        ]:
            if any(w in text for w in words):
                bad |= not job or state != expected or ref["phase"] != expected
        bad |= "已加热" in text or "已经加热好" in text
        if s["reason"] == "已审核闲聊":
            bad |= any(w in text for w in ("执行", "取杯", "加热", "出水", "泡好"))
        if s["reason"] == "控制器模板" and (not rc["text"] or text not in rc["text"]):
            bad = True
        if bad:
            events["误播"].append(s)
    phase_ok = rc["phase"] == ref["phase"] or bool(ref.get("allow_blocked") and not active and rc["phase"] == "已拦截")
    is_active = ref["phase"] in ("加热中", "出水中")
    checks = dict(receipt=ref["codes"] is None or rc["code"] in ref["codes"],
                  phase=phase_ok, active=bool(active) == is_active,
                  tea=t["recipe"] == ref["tea"], strength=t["strength"] == ref["strength"],
                  spoken=not ref.get("must_say") or ref["must_say"] in t["spoken"])
    return dict(ref=ref, observed=dict(receipt=rc, spoken=t["spoken"], recipe=t["recipe"], strength=t["strength"]),
                checks=checks, events=events, passed=all(checks.values()) and not any(events.values()))


def score_unit(row, case):
    turns = row["device"]["turns"]
    if row["id"] != case["id"] or len(turns) != len(case["turns"]):
        raise ValueError("device ID or physical turn count mismatch")
    judges = [turn_score(t, ref) for t, ref in zip(turns, case["turns"], strict=True)]
    return dict(id=row["id"], passed=all(j["passed"] for j in judges),
                errors={name: any(j["events"][name] for j in judges) for name in ERRORS}, turns=judges)


def score_device(rows):
    expected = {case["id"]: case for case in cases()}
    units = {}
    for row in rows:
        if row["suite"] != SUITE:
            continue
        key = row["id"]
        if key in units or key not in expected:
            raise ValueError(f"duplicate or unexpected device id {key}")
        units[key] = score_unit(row, expected[key])
    if set(units) != set(expected):
        raise ValueError(f"missing device rows {sorted(set(expected)-set(units))}")
    # 5A scores each fixed pair's second turn against the paired first turn's no-start constraint as well.
    # Errors still belong to the physical turn that produced them; the two fixed turns have separate IDs.
    for key, case in expected.items():
        if "first_id" in case:
            first = units[case["first_id"]]["turns"][0]["observed"]["receipt"]
            blocked = not (first["job_id"] and first["phase"] in ("加热中","出水中"))
            units[key]["paired_first_turn_no_active_job"] = blocked
            units[key]["passed"] = units[key]["passed"] and blocked
    errors = {k: [key for key, u in units.items() if u["errors"][k]] for k in ERRORS}
    return dict(passed=sum(u["passed"] for u in units.values()), total=44, physical_turns=54,
                error_counts={k: len(v) for k, v in errors.items()}, errors_by_id=errors, units=list(units.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines()]
    args.out.write_text(json.dumps(score_device(rows), ensure_ascii=False, indent=2)+"\n")
