"""泡茶机参考控制器（纯软件，无硬件 I/O）。

每台设备/交互会话持有一个实例。turn 接收用户原话、未受信任的模型提议及本次传感器快照。
update_sensors 和 controller_event 是可信设备侧的入口，不能作为模型工具暴露。
安全状态由固件信号给出；本模块不推测温度或加热完成时间。
"""
from dataclasses import dataclass
import json
import re
from threading import RLock
from types import MappingProxyType
from uuid import uuid4
from time import monotonic

DEFAULT_MENU = ("铁观音", "凤凰单丛")
STRENGTHS = ("淡", "标准")
FAULTS = {
    "no_cup": "未检测到杯子，已禁止出水。请放好杯子后重新发令。",
    "low_water": "水量不足，当前无法冲泡。请补足水后重新发令。",
    "lid_open": "盖子未合，当前无法冲泡。请合好盖子后重新发令。",
    "temperature_fault": "水温异常，已禁止加热和出水。请按设备提示检查。",
    "sensor_unknown": "设备状态未确认，已禁止加热和出水。请检查设备。",
}

@dataclass(frozen=True)
class Sensors:
    """当前固件的四个已确认安全位。新旧帧有效性应由设备驱动检查。"""
    cup: bool | None
    water: bool | None
    lid: bool | None
    temperature_ok: bool | None

    def fault(self):
        for value, code in ((self.temperature_ok, "temperature_fault"),
                            (self.lid, "lid_open"), (self.cup, "no_cup"), (self.water, "low_water")):
            if type(value) is not bool:
                return "sensor_unknown"
            if not value:
                return code
        return None

@dataclass(frozen=True)
class Receipt:
    code: str
    phase: str
    job_id: str | None
    text: str | None

    @property
    def active_job_id(self):
        return self.job_id if self.phase in ("加热中", "出水中") else None


# 可自由组合的闲聊句均不陈述设备进度；硬件事实只从控制器模板生成。
CHAT_TOPICS = (
    (("单丛这个名字", "单丛的名字"), ("单丛的“丛”指茶树。", "你更喜欢花香，还是果香？")),
    (("摇青", "做青"), ("做青时，翻动和静置交替，叶片的水分分布和香气会变化。",)),
    (("蜜兰", "花香"), ("香型是在描述闻起来像什么。", "你闻起来更像哪种花？")),
    (("午休",), ("今天的午休还剩多久？", "想安静一会儿也挺好。")),
    (("工作群",), ("工作群是突然安静了，还是消息太多？",)),
    (("湿鞋",), ("雨天双响。",)),
    (("咖啡因", "睡前"), ("想避开咖啡因，温水是一个选择。",)),
    (("午餐", "吃饭"), ("好，先聊午餐。", "今天想吃热乎的还是清爽的？")),
    (("累", "加班"), ("听起来今天挺累的。", "你想吐槽哪一段？")),
    (("下雨", "听雨"), ("听雨。", "你那边雨声大吗？")),
    (("先不", "别启动", "到点"), ("好，先聊会儿。",)),
    (("聊茶","茶香","茶味","乌龙","铁观音","凤凰单丛"), ("茶的香气有时会随温度变化。", "你平时喜欢什么香气？")),
    ((), ("你接着说，我听着呢。", "嗯，我在听。", "后来呢？")),
)
SAFE_CHAT = {line.rstrip("。？！") for _, lines in CHAT_TOPICS for line in lines}
# 这组词能组成开放的情绪、闲聊和感官问句，不含工序、命令或完成语义。
# 可直接保留模型自由组合的句子。分词必须覆盖整句，没有通配符。
SOCIAL_WORDS = frozenset("""
我 你 他 她 我们 你们 他们 她们 自己 大家 同事 老板 朋友 室友 家人 同桌 邻居 猫 猫咪 小狗
工作 加班 午休 周一 周末 午餐 早餐 晚饭 消息 群聊 日子 故事 音乐 歌 电影 笑话 书
今天 昨天 明天 今晚 最近 平时 有时 有时候 现在 刚才 当时 每天 突然 其实 有点 有些 特别 挺
真的 确实 好像 似乎 也许 大概 可能 总是 更 最 太 很 还 也 都 又 先 光 只 又是 一直
忙 辛苦 麻烦 难受 累 疲惫 委屈 委屈的 开心 高兴 快乐 轻松 紧张 安静 无聊 好奇 有趣
不着急 不容易 不舒服 不想 不太 不用 不必 不 也没 没 什么 哪个 哪种 哪一段 谁 怎么 为什么
是 不是 有 想 想要 觉得 喜欢 惦记 在意 听 听着 听听 听起来 说 说说 聊 聊聊 讲 讲讲
吐槽 发呆 休息 歇一会儿 陪 陪着 笑 更像 像 看 看看 分享 呢 吗 么 了 呀 啊 吧 哦 嗯 哈哈
的 得 地 这 那 这个 那个 一下 一会儿 一点 一点儿 两句 一句 一件事 哪件事 后来 然后
和 或者 还是 但是 因为 所以 如果 下雨 雨声 下班 上班 办公室 窗外 晚霞 声音 心情 感觉
花香 果香 兰花香 香气 蜜香 闻起来 闻 甜香 草木 坚果 花 果子 哪一款 哪一种 口味
""".split())
def safe_social(sentence):
    # 长度有界，动态分词避免有歧义词前缀时反复回溯。
    value=re.sub(r"[ ，,。？！!?；;、…]","",sentence)
    if not value or len(value)>180:
        return False
    reachable={0}
    for i in range(len(value)):
        if i in reachable:
            reachable.update(i+n for n in range(1,min(6,len(value)-i)+1)
                             if value[i:i+n] in SOCIAL_WORDS)
    return len(value) in reachable


BOUNDARIES = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]*")
PROGRESS_WORDS = re.compile(r"泡好|泡完|完成|好了|好啦|加热|热好|热完|烧开|烧好|沸腾|水开|出水|注水|倒水|出汤|出茶|取杯|端走|取走|开泡|冲好|冲完|制好|制作|就绪|可以喝|流出|流进")
def chat_options(user):
    for triggers, lines in CHAT_TOPICS:
        if not triggers or any(t in user for t in triggers):
            return lines


def parse_proposal(raw):
    def unique(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("duplicate field")
        return dict(pairs)
    try:
        p = json.loads(raw, object_pairs_hook=unique) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if type(p) is not dict or set(p) != {"action", "tea", "strength"}:
        return None
    if p["action"] == "start":
        if type(p["tea"]) is str and p["strength"] in STRENGTHS:
            return p
    elif p["action"] in ("chat","clarify","stop","status") and p["tea"] is None and p["strength"] is None:
        return p
    return None

class TeaController:
    def __init__(self, menu=DEFAULT_MENU, *, recipe_strengths=None):
        if isinstance(menu,str): raise ValueError("菜单需要独立茶名")
        self.menu = tuple(menu)
        if not self.menu or len(set(self.menu)) != len(self.menu) or not all(isinstance(t,str) and t.strip()==t and t for t in self.menu):
            raise ValueError("菜单需要非空、唯一的茶名")
        configured = {t: STRENGTHS for t in self.menu} if recipe_strengths is None else recipe_strengths
        if set(configured) != set(self.menu):
            raise ValueError("每款菜单茶都需要配方能力")
        modes = {t: tuple(configured[t]) for t in self.menu}
        if any(not v or len(set(v)) != len(v) or any(w not in STRENGTHS for w in v) for v in modes.values()):
            raise ValueError("配方浓度必须来自本机支持的档位")
        self.recipe_strengths = MappingProxyType(modes)
        self._job_session = None
        self._completed_job = None
        self._completed_recipe = None
        self._strength = None
        self._tea_pattern = "(?:" + "|".join(re.escape(t) for t in self.menu) + ")"
        tokens = ("请现在","帮我","替我","给我","麻烦","这杯","这一杯","这次","这回","今天",
                      "就","想喝","我想喝","我要","我点","我","用","按","选","要","浓度","浓淡",
                      "口味","档","的","一杯","一份","现泡","现冲","冲泡","开泡","开机","启动",
                      "执行","开始","请","现在","马上","立即","泡","冲","沏","做","来","标准","淡",
                      "一点","点","吧","啦")
        pieces = "|".join(re.escape(v) for v in sorted((*self.menu,*tokens), key=len,reverse=True))
        self._command_token = re.compile(pieces)
        self._menu_text = "、".join(self.menu)
        self._registered = None
        self._excluded = set()
        self._pending = None
        self._pending_strength = None
        self._job = None
        self._phase = "空闲"
        self._recipe = None
        self._fault = None
        self._lock = RLock()
        self._focus = None
        self._last_sequence = -1
        self._pending_session = None

    def _receipt(self, code, text):
        return Receipt(code,self._phase,self._job or self._completed_job,text)

    def _clear(self, phase):
        self._completed_job=self._completed_recipe=self._strength=self._job_session=None
        self._forget_choice()
        self._phase=phase
        self._job=self._recipe=self._pending=self._pending_strength=self._pending_session=None

    def _apply_sensors(self, sensors):
        code=sensors.fault() if isinstance(sensors,Sensors) else "sensor_unknown"
        if code:
            self._clear("已拦截")
            self._fault=code
        elif self._phase=="已拦截" and self._fault is None:
            self._phase="空闲"
        return code

    def update_sensors(self, sensors):
        """异步传感器入口。故障即中止任务；恢复传感器不会重启。"""
        with self._lock:
            self._apply_sensors(sensors)
            return self._status()

    def controller_event(self, job_id, code, sensors):
        """由控制器驱动调用。只有同一有效任务的完成回执能播报完成。"""
        with self._lock:
            if self._apply_sensors(sensors):
                return self._status()
            if not isinstance(code,str):
                code="sensor_unknown"
            if code in FAULTS:
                self._clear("已拦截")
                self._fault=code
                return self._status()
            if code=="fault_cleared":
                self._fault=None
                if self._phase=="已拦截":self._clear("空闲")
                return self._status()
            if code not in ("dispensing","completed","stopped"):
                self._fault="sensor_unknown"
                self._clear("已拦截")
                return self._status()
            if self._fault or self._job is None or job_id!=self._job:
                return self._receipt("stale","该回执与当前任务不匹配，未更改状态。")
            if code=="stopped":
                self._clear("已停止")
                return self._receipt("stopped","已停止，当前没有冲泡任务。")
            if code=="dispensing":
                self._phase="出水中"
            else:
                finished, recipe = self._job, self._recipe
                self._clear("已完成")
                self._completed_job, self._completed_recipe = finished, recipe
            return self._status()

    def _status(self):
        if self._fault:
            return self._receipt(self._fault,FAULTS[self._fault])
        if self._phase=="已完成" and self._completed_job:
            return self._receipt("completed",f"控制器已确认这杯{self._completed_recipe}泡茶完成，请小心取杯。")
        if self._phase=="已完成":
            return self._receipt("unverified","当前没有可核实的完成回执。")
        if self._phase in ("加热中","出水中"):
            return self._receipt("status",f"{self._recipe}当前{self._phase}。")
        return self._receipt("status",f"设备当前{self._phase}。")

    def _ask(self):
        preference = f"已记下{self._pending_strength}档。" if self._pending_strength else ""
        return self._receipt("choose_tea",f"{preference}本机只支持{self._menu_text}。这杯想选哪一种？本次没有启动新任务。")

    _register_verb = r"(?:暂定|暂选|预选|先选|选定|圈(?:下|上|定|起来)?|登记|记(?:下|住|着)|留(?:下|着)|保留|暂存|备选|改选|改成|换成)"
    _exclude_verb = r"(?:划掉|排除|去掉|划去|不要|不选|不考虑|跳过)"
    _choice_change = re.compile(_register_verb + "|" + _exclude_verb +
                                r"|不|没|别|放弃|取消|换|改|撤|还是|自带|重来|算了|忘掉")

    def _forget_choice(self):
        self._registered = None
        self._excluded.clear()
        self._pending = self._pending_strength = self._focus = None

    def _selection_intent(self, utterance, previous, remembered_strength, focus):
        """完整解析用户自己的分句。只有整轮语义一致才提交登记或排除。"""
        clauses = [s for s in re.split(r"[，,。.!！；;、]+", re.sub(r"\s+", "", utterance)) if s]
        if not clauses or len(clauses) > 8 or any(len(c) > 128 for c in clauses):
            return None
        selected = previous or self._registered
        excluded = self._excluded.copy()
        strength = None
        order = None
        confirmed = unresolved = False
        changed = deferred = chose = False
        menu_name = self._tea_pattern
        register = self._register_verb
        subject = r"(?:我)?(?:今天|这杯|这次)?(?:给自己|给我|帮我|替我)?(?:先)?(?:把|将)?"
        noun = r"(?:款|样|种|个|杯)(?:茶)?"
        recorded = rf"(?:刚才?|之前|前面)?(?:我)?(?:{register}|选|记|留)(?:过|下|好)?的(?:[这那]一?{noun})?"
        demonstrative = rf"(?:就)?(?:[这那]一?{noun}|它)"
        other = rf"(?:另(?:外)?一{noun}|剩(?:下|余)(?:的)?(?:[那这]一?{noun}|一?{noun})?)"
        reference = rf"(?:{recorded}|{demonstrative}|{other})"
        serve = r"(?:开泡|启动|开始(?:冲泡)?|开机|下(?:单|一杯)|冲泡|泡|做|冲|沏|来)(?:一杯|一份|杯)?(?:吧)?"
        now = r"(?:现在|马上|立即)?"
        degree = r"(?:清淡|淡(?:一点|点)?|标准)(?:档|浓度|口味|的)?"

        def resolve(value):
            if value in self.menu:
                return value
            if re.fullmatch(other, value):
                available = set(self.menu) - excluded
                if selected:
                    available.discard(selected)
                return next(iter(available)) if len(available) == 1 and (selected or excluded) else None
            if re.fullmatch(recorded, value):
                return selected
            return selected or focus

        consumed = -1
        for i, clause in enumerate(clauses):
            if i == consumed:
                continue
            # 限定可忽略的附带说明。未知尾句、否定、预约或疑问均拒绝整轮授权。
            if re.fullmatch(r"(?:茶单|菜单)(?:先)?(?:收起|收好|合上)(?:来|吧|了)?", clause):
                continue
            if re.fullmatch(r"(?:但)?(?:我)?(?:先|暂时|还)?(?:不(?:用|要)?(?:泡|启动|开机|下单)|没(?:有)?打算(?:喝|泡))", clause):
                deferred = True
                continue
            if re.fullmatch(rf"(?:就|要|按|用|选)?{degree}", clause):
                new_strength = "淡" if "淡" in clause else "标准"
                if strength and strength != new_strength:
                    return None
                strength = new_strength
                continue
            description = re.fullmatch(rf"(?:另(?:外)?一{noun})是({menu_name})", clause)
            if description and selected and description[1] != selected and changed:
                continue
            choice = re.fullmatch(rf"{subject}{register}(?:为|的是)?({menu_name}|{reference})(?:吧)?", clause)
            if not choice:
                choice = re.fullmatch(rf"{subject}({menu_name})(?:给我|帮我)?(?:先)?{register}(?:一下|吧)?", clause)
            if not choice:
                choice = re.fullmatch(rf"(?:我)?(?:刚才?|已经)(?:{register})的(?:茶)?是({menu_name})", clause)
            if choice and order is None:
                if chose and not re.search(r"改选|改成|换成", clause):
                    return None
                chose = True
                selected = resolve(choice[1])
                if not selected:
                    return None
                excluded.discard(selected)
                changed = True
                continue
            exclude = self._exclude_verb
            removal = re.fullmatch(rf"{subject}{exclude}({menu_name}|{reference})", clause)
            if not removal:
                removal = re.fullmatch(rf"{subject}({menu_name}|{reference})(?:也|先)?{exclude}(?:了)?", clause)
            if removal and order is None:
                tea = resolve(removal[1])
                if not tea:
                    return None
                excluded.add(tea)
                if tea == selected:
                    selected = None
                changed = True
                continue
            if re.fullmatch(reference, clause) and order is None and i+1 < len(clauses):
                # 指代语与执行动词可以分在相邻两句；只有完整的即时命令才授权。
                if re.fullmatch(rf"{now}{serve}", clauses[i+1]):
                    consumed = i + 1
                    order = resolve(clause)
                    if not order or order in excluded:
                        unresolved = True
                    continue
            if re.fullmatch(rf"(?:确认|请)?{now}(?:开泡|启动|开始)(?:吧)?", clause):
                if confirmed:
                    return None
                confirmed = True
                if order is None:
                    # 直接确认仅承接本轮选择、相邻选择，或确有排除记录的唯一菜单项。
                    remaining = set(self.menu) - excluded
                    if changed or previous:
                        order = selected
                    if not order and excluded and len(remaining) == 1:
                        order = next(iter(remaining))
                    if not order:
                        unresolved = True
                continue
            request = re.fullmatch(rf"(?:就|请|麻烦|帮我|给我)?(?:按|用)?({reference}){now}{serve}", clause)
            if not request:
                request = re.fullmatch(rf"(?:就|请|帮我|给我)?{now}{serve}({reference})", clause)
            if request and order is None:
                target = resolve(request[1])
                if not target or target in excluded:
                    unresolved = True
                order = target
                continue
            return None

        if not (order or unresolved or changed) or (order and deferred):
            return None
        self._excluded = excluded
        self._registered = selected
        if unresolved:
            return "unresolved", None, None
        if order:
            return "start", order, strength or remembered_strength or "标准"
        return "select" if selected else "excluded", selected, strength or remembered_strength

    def _authorize(self, text, previous, remembered_strength):
        # 完整句法匹配，助手说过的选项、转述、疑问和陌生词均不能授予执行权限。
        t=self._tea_pattern
        select=re.fullmatch(fr"(?:我(?:这杯|今天)?(?:明确)?选|我选择|这杯我选)({t})(?:先不(?:泡|启动))?",text)
        if select:
            return "select",select[1],None
        specs="(?:淡一点|淡点|淡口|淡的|淡泡|淡|标准浓度|标准口味|标准)"
        tea_match=re.fullmatch(fr"(?:请|麻烦|现在|今天)?(?:现在)?(?:帮我|给我|替我)?(?:现冲|冲泡|冲|泡|做|来)(?:一杯|杯|一份)?(?:淡的|淡口的|淡泡的|标准浓度的)?({t})(?:现在)?(?:立即)?(?:开始吧|开始|启动|开泡|泡吧)?(?:要|按|选)?(?:{specs})?(?:档)?",text)
        if not tea_match:
            tea_match=re.fullmatch(fr"(?:我选|我点|确认选择)?({t})(?:现在|马上|立即)?(?:开泡|泡吧|开始吧|开始|启动)(?:要|按|选)?(?:{specs})?",text)
        if not tea_match:
            tea_match=re.fullmatch(fr"(?:今天的第一杯就喝|今天这杯就选|这杯我选)({t})(?:麻烦|请)?(?:现在|马上)?(?:开泡|开始|启动)(?:要)?(?:{specs})?",text)
        if not tea_match:
            tea_match=re.fullmatch(fr"(?:再开|开|帮我冲)(?:一杯|杯)(?:淡的|标准浓度的)?({t})(?:马上开始|现在开始|现在泡吧)?",text)
        name=tea_match[1] if tea_match else None
        if name is None:
            # 只接受单一配方、单一浓度及封闭词表中的即时命令；疑问、否定、
            # 条件、转述和时间预约均无法匹配。无法确定时退回确认。
            names=[v for v in self.menu if text.count(v)]
            direct=re.search(r"(?:请|帮我|替我|给我|麻烦|马上|现在|立即)(?:现在|马上)?(?:开泡|开机|启动|冲泡|开始|泡|冲|沏|做|执行)",text)
            if (len(names)==1 and text.count(names[0])==1 and direct and
                    len(text)<=128 and self._command_token.sub("",text)=="" and not ("标准" in text and "淡" in text)):
                name=names[0]
        if name is None and previous and re.fullmatch(r"(?:就这个|就它|就按我刚选的|用我刚选的那款|确认)(?:现在|马上)?(?:启动|开泡|开始吧|开始|泡吧)(?:要|按)?(?:标准|淡一点|淡)?",text):
            name=previous
        if name:
            strength="淡" if "淡" in text else "标准" if "标准" in text else remembered_strength or "标准"
            return "start",name,strength
        return None,None,None

    def turn(self, user_text, raw_proposal, sensors, *, sequence=None, session="local"):
        """唯一的模拟启动入口；成功时返回 job_id，供设备事件匹配。"""
        with self._lock:
            text = re.sub(r"[\s，,。.!！、：:]", "", user_text) if isinstance(user_text,str) else ""
            self._apply_sensors(sensors)
            if sequence is not None:
                if type(sequence) is not int or sequence<0 or sequence<=self._last_sequence:
                    return self._status() if self._fault else self._receipt("duplicate","这条指令已过期或序号无效，没有执行新动作。")
                self._last_sequence=sequence
            if self._fault:
                return self._status()
            same_session = isinstance(session, str) and bool(session) and session == self._pending_session
            if not same_session:
                self._forget_choice()
            if re.search(r"没(?:有)?(?:点|选|定)|尚未(?:选|点)|没(?:有)?决定|没想好|待定|未选|犹豫|纠结|上一位|上位客人|换人|下一位|轮到我|该我了|换自己的杯|先聊别的|重新选|不算数|作废", text):
                self._forget_choice()
            # 能力查询保留前序选择，永远不会触发执行或修改浓度。
            capability = self._strength_question(text, session)
            if capability:
                return self._receipt("capability", capability)
            previous, strength=(self._pending,self._pending_strength) if same_session else (None,None)
            self._pending = None
            if not self._registered:
                self._pending_strength = None
            self._pending_session=session
            # 话题只保留给同一会话的下一轮；普通选择与话题焦点分别存放。
            focus=self._focus
            self._focus=None
            explicit_stop = re.fullmatch(
                r"(?:请|快|现在|马上)?(?:停止|停下|停|停一下|取消|取消任务|停止加热|停止冲泡|停机|停止出水)(?:吧|这一杯|当前任务)?"
                r"|(?:先)?(?:别泡了|别出水了|别加热了|别再泡了|停一下吧|停机吧)|先别继续加热了现在把任务取消"
                r"|快停别再往杯里倒了|等等先别出水停一下|把正在泡的那杯停掉|这杯先取消|先停|别再给我加热了停机吧|请结束当前冲泡现在停止", text)
            if explicit_stop:
                self._clear("已停止")
                return self._receipt("stopped","已停止，当前没有冲泡任务。")
            # 用户只提到一款并表示喜欢：明确回读该茶，不添加另一候选，不授予动作权限。
            topic=re.fullmatch(fr"({self._tea_pattern})(?:听起来|闻起来|看起来)(?:不错|挺不错|很不错|挺好|很合我口味|挺合我口味)",text)
            if topic and self._job is None and isinstance(session,str) and session:
                self._focus=(session,topic[1],monotonic())
                return self._receipt("chat",f"你说的是{topic[1]}。这杯还没启动。")
            quoting=re.fullmatch(r"(?:同事|朋友|家里人|室友).{0,28}[“‘「].{1,24}[”’」](?:我)?(?:只是|只是在)(?:学.{0,6}说话|转述|引用|照着念)",text)
            if quoting:
                return self._receipt("chat","明白，你是在转述，当前没有执行新动作。")
            # 明确拒绝替换的用户无需再次被要求换茶；即使模型建议改泡也不能执行。
            if re.search(r"其他茶都?别给我做|不接受替换|不要替代|不行就算了|不要换别的|只想用自己带的|只用自带|只用自己带的|只认我这包|不换别的",text):
                self._forget_choice()
                return self._receipt("no_substitution", f"本机可选{self._menu_text}。本次没有启动，也不会换成其他茶。")
            capability = re.search(r"(?:能不能|可不可以)(?:帮我)?(?:冲泡|泡|做|冲|沏)|(?:能|可以|支持)(?:帮我)?(?:冲泡|泡|做|冲|沏).{0,16}(?:吗|么|[?？])|"
                                   r"(?:这台|本机|机器|你)(?:能|可以|支持)(?:泡|做)?(?:哪些|哪几种|什么)茶|"
                                   r"(?:有什么茶|茶种菜单|支持的茶种|完整菜单)", text)
            if capability:
                return self._receipt("menu",f"本机支持{self._menu_text}。菜单以外的茶当前做不了，本次没有启动新任务。")
            p=parse_proposal(raw_proposal)
            mode,tea,w=self._authorize(text,previous,strength)
            if mode is None and isinstance(session, str) and session and not self._job:
                candidate = focus[1] if focus and focus[0] == session and monotonic()-focus[2] <= 90 else None
                selection = self._selection_intent(user_text if isinstance(user_text, str) else "", previous, strength, candidate)
                if selection:
                    mode,tea,w = selection
                elif self._choice_change.search(text) or any(t in text for t in self.menu):
                    self._forget_choice()
            simple_order=re.fullmatch(r"(?:请|麻烦)?(?:给我|替我|帮我)?(?:泡|冲泡|现冲|沏|现泡)(?:一杯|一份|杯)?([一-龥]{2,10})",text)
            if mode is None and simple_order and all(t not in simple_order[1] for t in self.menu):
                self._forget_choice()
                if "淡" in text:self._pending_strength="淡"
                return self._ask()
            if mode == "unresolved":
                return self._ask()
            if mode == "excluded":
                remaining = [t for t in self.menu if t not in self._excluded]
                if remaining:
                    return self._receipt("choose_tea",f"已记下排除项，可选{'、'.join(remaining)}。这杯还没启动。")
                return self._ask()
            reference=re.fullmatch(r"(?:就这个|就这款|就它|刚才这款)(?:给我|帮我|替我)?(?:冲|泡)?(?:一杯|杯)?(?:淡一点的|淡点的|淡的|标准的|标准浓度)?(?:现在|马上|立即)?(?:开始吧|开始|开泡|启动|泡吧)",text)
            if (mode is None and focus and session==focus[0] and monotonic()-focus[2]<=90
                    and reference and self._job is None):
                # 下文仍会独立检查模型给出的 start、合法菜单、茶种、浓度及所有联锁。
                mode,tea,w="start",focus[1],"淡" if "淡" in text else "标准"
            if mode=="select":
                if self._phase=="已完成":self._clear("空闲")
                self._pending_session=session
                self._pending=tea
                self._registered=tea
                self._excluded.discard(tea)
                self._pending_strength=w or strength
                return self._receipt("selected",f"已选{tea}，等你确认开始。")
            if p is None:
                return self._receipt("invalid",f"未收到有效指令，本次没有启动。当前茶种有{self._menu_text}。")
            if p["action"]=="start" or mode=="start":
                if (mode!="start" or p["action"]!="start" or tea not in self.menu or
                        p["tea"]!=tea or p["strength"]!=w):
                    if not tea and p["tea"] not in self.menu and p["tea"] is not None:
                        return self._receipt("unsupported",f"请求的茶种不在本机菜单。当前可选{self._menu_text}，本次没有启动。")
                    if not tea and "淡" in text:self._pending_strength="淡"
                    return self._ask()
                if self._job is not None:
                    return self._receipt("busy",f"{self._recipe}仍在冲泡，本次没有启动新任务。")
                if w not in self.recipe_strengths[tea]:
                    return self._receipt("unsupported_strength", f"{tea}只支持{'、'.join(self.recipe_strengths[tea])}档。这杯没有启动。")
                self._completed_job=self._completed_recipe=None
                self._forget_choice()
                self._phase="加热中";self._recipe=tea;self._strength=w;self._job=uuid4().hex
                self._job_session=session
                return self._receipt("started",f"{tea}已按{w}浓度开始加热。")
            wants_brew=re.search(r"(?:给我|帮我|替我|麻烦你|请你).*(?:冲|泡|沏|来杯|开一杯)|(?:冲|泡|沏|来|开)(?:一杯|杯|壶)|做一杯|开泡|菜单|可选茶种",text)
            if wants_brew and not re.search("怎么|为什么|区别|什么工艺|不用做|先不泡",text):
                if "淡" in text:self._pending_strength="淡"
                return self._ask()
            if p["action"]=="status" or re.search("好了没|好了吗|泡好了|泡好没|能取杯|取杯了|设备状态|机器状态|到哪一步|还在泡|还有多久|机器报错",text):
                return self._status()
            if p["action"]=="stop":
                return self._receipt("confirm_stop","请确认是否要停止当前任务。")
            if p["action"]=="clarify":
                if "淡" in text:self._pending_strength="淡"
                return self._ask()
            return self._receipt("chat",None)


    def _strength_question(self, text, session):
        tuning = re.search(r"(?:调|改|设|选|泡得|做得)(?:成|到|得|为)?(?:更|清|偏)?淡|淡(?:档|口味)|调(?:节)?浓(?:淡|度)|(?:能不能|可以|能否|可不可以)(?:泡|做)(?:得)?淡", text)
        question = re.search(r"能不能|可不可以|能否|是否|可以|支持|[吗么？?]|只问|能调", text)
        if not tuning or not question:
            return None
        # 多个茶名或未知专名均不借用上一款，避免把能力张冠李戴。
        # 只把封闭的问法剥去；残留未知名字时不能套用上次选择。
        clause = re.split(r"[？?]|我只问|我只是问|我就问", text, maxsplit=1)[0] or text
        residue = re.sub(r"可不可以|能不能|是不是|能否|是否|可以|支持|请问(?:一下)?|我想问(?:一下)?|只问|能", "",clause)
        residue = re.sub(tuning.re.pattern, "",residue)
        residue = re.sub(r"^(?:这台机器|本机|帮我|替我|把|给)|(?:一点儿?|一些|一档|点儿?|吗|么|呢|呀|[?？])+$", "",residue)
        residue = re.sub(r"^(?:把|给)", "",residue)
        names = [t for t in self.menu if t in clause]
        refs = r"(?:我选的|刚才选的|刚选的|刚才的|刚才|上次选的|前面选的|已选的|这款茶|那款茶|这款|那款|这杯茶|这杯|那杯|这个|那个|它|的浓度|的配方|的|茶叶)*"
        if len(names)==1 and re.fullmatch(refs+re.escape(names[0])+refs,residue):
            tea = names[0]
        elif not names and re.fullmatch(refs,residue):
            tea = (self._pending or self._registered) if session and session==self._pending_session else None
            if not tea and self._focus and self._focus[0]==session and monotonic()-self._focus[2]<=90:
                tea = self._focus[1]
            if not tea and self._job and self._job_session==session:
                tea = self._recipe
        else:
            tea = None
        if not tea and not names and re.fullmatch(r"[\u3400-\u9fff]{2,12}",residue) and not re.search("这|那|刚|自己|当前|上|我|茶种|还没|哪|两款|两种|两个|每款|每种|都|是否|谁",residue):
            return f"本机没有{residue}的配方，无法为它调淡。当前菜单是{self._menu_text}。"
        if not tea:
            return f"本机菜单是{self._menu_text}。你想查哪款的浓度档位？"
        modes = self.recipe_strengths[tea]
        if "淡" in modes:
            return f"可以，{tea}的配方支持淡档，启动前可选。"+("已经运行的这杯不能中途改档。" if self._job and tea==self._recipe else "")
        return f"{tea}的配方只支持{'、'.join(modes)}档，调不了淡。"

    def _state_note(self, session):
        state = self._status()
        if state.active_job_id:
            job = f"当前任务编号: {state.job_id} | 执行茶种: {self._recipe} | 浓度: {self._strength}"
        elif state.code == "completed" and state.job_id:
            job = f"当前无运行任务 | 完成回执编号: {state.job_id} | 完成茶种: {self._completed_recipe}"
        else:
            job = "当前无任务"
        parts = [state.phase, job, f"本机菜单仅限: {self._menu_text}"]
        if self._fault:
            parts.append("故障: "+self._fault)
        if session and session == self._pending_session:
            if self._pending or self._registered:
                parts.append("本轮前已选: "+(self._pending or self._registered))
            if self._excluded:
                parts.append("已排除: "+"、".join(t for t in self.menu if t in self._excluded))
        if self._pending_strength and session and session==self._pending_session:
            parts.append("用户已选浓度: "+self._pending_strength+"（沿用到下一次确认）")
        if self._focus and self._focus[0]==session and monotonic()-self._focus[2]<=90:
            parts.append("上一轮提及: "+self._focus[1])
        return "[实时状态: "+" | ".join(parts)+"]"

    def _model_input(self, system, payload, session):
        # 原话留在 JSON 中，可信状态只由程序另写到系统消息尾部。
        return [dict(role="system",content=system+"\n"+self._state_note(session)),
                dict(role="user",content=json.dumps(payload,ensure_ascii=False))]

    def _deliver(self, ticket, draft, user, emit, audit, read_sensors=None):
        # 未审核的任意自然语言不能自行证明与硬件无关。进度句和无法认证的句子
        # 均不原样播；已审核闲聊可逐句保留，模型不能靠换一种说法绕过出口。
        controller_text = ticket.text is not None
        source = ticket.text if controller_text else draft
        source = source if isinstance(source,str) else ""
        chunks = BOUNDARIES.findall(source.strip()) or [""]
        spoken = []
        delivered = ticket
        for sentence in chunks:
            with self._lock:
                if read_sensors is not None:
                    self._apply_sensors(read_sensors())
                latest = self._status()
                stale = (ticket.phase,ticket.job_id)!=(latest.phase,latest.job_id) or (
                    self._fault is not None and ticket.code!=self._fault)
                end = stale
                if stale:
                    delivered, line, reason = latest, latest.text, "回执已变化"
                elif controller_text:
                    line, reason = sentence, "控制器模板"
                elif sentence.strip().rstrip("。？！!?；;") in SAFE_CHAT or safe_social(sentence):
                    line, reason = sentence.strip(), "已审核闲聊"
                elif PROGRESS_WORDS.search(sentence):
                    line, reason = latest.text, "改播最新回执"
                else:
                    line, reason = chat_options(user)[0], "改用已审核闲聊"
                if not spoken or spoken[-1]!=line:
                    if emit is not None:
                        emit(line)  # 同步、非阻塞交付；队列消费者须在实际交付时运行本检查。
                    spoken.append(line)
                audit.append(dict(source=sentence,spoken=line,reason=reason,phase=latest.phase,
                                  job_id=latest.job_id,controller_code=latest.code))
            if end:
                break
        return delivered,"".join(spoken)

    def respond(self, user_text, proposal, sensors, chat, *, sequence=None, session="local", emit=None):
        """兼容已有提议的调用；生成新提议的完整产品入口为 run_turn。"""
        ticket = self.turn(user_text,proposal,sensors,sequence=sequence,session=session)
        draft = None
        if ticket.text is None:
            with self._lock:
                prompt = self._state_note(session)+"\n"+user_text
            draft = chat(prompt)
        return self._deliver(ticket,draft,user_text,emit,[])

    def run_turn(self, user, history, infer, sensors, *, plan_prompt, reply_prompt,
                 session="local", sequence=None, emit=None):
        """infer(messages, token_limit) 只负责推理。sensors() 在推理前后读取最新硬件快照。"""
        with self._lock:
            before = self.update_sensors(sensors())
            planning = self._model_input(plan_prompt,dict(历史=history,用户原话=user,设备阶段=before.phase),session)
        proposal = infer(planning,96)
        with self._lock:
            ticket = self.turn(user,proposal,sensors(),sequence=sequence,session=session)
            replying = None
            if ticket.text is None:
                receipt = dict(结果="未请求动作",状态=dict(阶段=ticket.phase,当前茶种=self._recipe,当前浓度=self._strength),
                               传感器事件=None,联锁=self._fault)
                options = chat_options(user)
                guidance = "\n自然地回应用户；设备进度取自回执。可参考："+ " ".join(options)
                replying = self._model_input(reply_prompt+guidance,dict(历史=history,用户原话=user,
                              解析结果=parse_proposal(proposal),回执=receipt),session)
        draft = infer(replying,128) if replying else None
        audit = []
        receipt, words = self._deliver(ticket,draft,user,emit,audit,sensors)
        trace = dict(plan_input=planning,proposal=proposal,reply_input=replying,draft=draft,sentences=audit)
        return receipt, words, trace
