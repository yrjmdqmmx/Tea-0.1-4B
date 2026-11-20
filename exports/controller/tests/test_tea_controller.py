"""python3 -m unittest discover -s 智能泡茶机/控制器"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import unittest
from tea_controller import DEFAULT_MENU, FAULTS, Sensors, TeaController, parse_proposal

OK=Sensors(True,True,True,True)
def plan(a,t=None,s=None):return dict(action=a,tea=t,strength=s)
P=plan("start","铁观音","标准")
T="现在给我泡杯铁观音"
class TestControl(unittest.TestCase):
    def setUp(self):self.c=TeaController()
    def no_chat(self,_): self.fail("关键回执不应送到模型")
    def test_faults_block_before_model(self):
        for attr,code in [('cup','no_cup'),('water','low_water'),('lid','lid_open'),('temperature_ok','temperature_fault')]:
            for a in (P,plan('chat'),None,plan('clarify')):
                with self.subTest(attr=attr,proposal=a):
                    c=TeaController()
                    receipt,words=c.respond(T,a,replace(OK,**{attr:False}),self.no_chat)
                    self.assertEqual((receipt.code,words,receipt.job_id),(code,FAULTS[code],None))
    def test_unknown_sensors(self):
        for bad in (None,replace(OK,lid=None),replace(OK,water=1),replace(OK,temperature_ok="yes")):
            self.assertEqual(TeaController().turn(T,P,bad).code,"sensor_unknown")
    def test_missing_or_hallucinated(self):
        for q in ["请给我泡杯茶","我同事喝铁观音，给我来杯淡的","假如我让你泡铁观音呢？","别给我泡铁观音",
                  "要么铁观音要么凤凰单丛，你看着办","铁观音换成龙井","泡一杯龙井","‘给我泡铁观音’是引用",
                  "泡铁观音？","别开始，给我泡杯铁观音只是例子","我喝铁观音只是昨天的事"]:
            for t in DEFAULT_MENU:
                with self.subTest(q=q,t=t):
                    out=self.c.turn(q,plan('start',t,'标准'),OK)
                    self.assertNotEqual(out.code,'started')
                    self.assertIsNone(out.job_id)
                    self.assertTrue(all(x in out.text for x in self.c.menu))
    def test_invalid_parameters(self):
        for p in (None,plan("start",None,"标准"),plan('start','凤凰单丛','浓'),plan('start','龙井','标准'),
                  {**P,'power':True},'{"action":"chat","action":"start","tea":"铁观音","strength":"标准"}'):
            self.assertNotEqual(self.c.turn(T,p,OK).code,'started')
        self.assertIsNone(parse_proposal('[]'))
    def test_selection_from_user_only(self):
        self.assertEqual(self.c.turn('你有哪两种？',plan('clarify'),OK).code,'choose_tea')
        self.assertNotEqual(self.c.turn('就按我刚选的，现在启动。',P,OK).code,'started')
        self.assertEqual(self.c.turn('我选铁观音。',plan('clarify'),OK).code,'selected')
        self.assertEqual(self.c.turn('就按我刚选的，现在启动。',P,OK).code,'started')
    def test_natural_requests_and_noncommands(self):
        for text,tea,strength in [("麻烦泡一杯凤凰单丛，标准浓度，现在开始。","凤凰单丛","标准"),
                                  ("我想喝铁观音，麻烦现在按淡档冲泡，马上开始。","铁观音","淡"),
                                  ("请用凤凰单丛给我做一杯，口味标准。","凤凰单丛","标准")]:
            c=TeaController()
            self.assertEqual(c.turn(text,plan("start",tea,strength),OK).code,"started")
        for text in ("明天请泡一杯铁观音","如果我说现在开始泡铁观音","请泡铁观音还是凤凰单丛",
                     "我想喝铁观音","等我开完会再帮我泡铁观音","他让我说请泡铁观音",
                     "请泡铁观音，先淡后标准","能请你帮我泡铁观音吗","请泡铁观音，先别启动"):
            self.assertIsNone(self.c.turn(text,P,OK).job_id)

    def test_stale_selection(self):
        self.c.turn('我选铁观音',plan('clarify'),OK)
        self.c.turn('我们先聊别的事',plan('chat'),OK)
        self.assertNotEqual(self.c.turn('确认开始',P,OK).code,'started')
    def test_do_not_substitute_or_overwrite_strength(self):
        for p in (plan('start','凤凰单丛','标准'),plan('start','铁观音','淡')):
            self.assertNotEqual(self.c.turn(T,p,OK).code,'started')
    def test_remember_strength_without_choosing_for_user(self):
        self.c.turn('来一杯淡的',plan('clarify'),OK)
        self.c.turn('我选铁观音',plan('clarify'),OK)
        self.assertEqual(self.c.turn('确认开始',plan('start','铁观音','淡'),OK).code,'started')
    def test_interrupt_and_no_autorestart(self):
        job=self.c.turn(T,P,OK).job_id
        self.assertIsNotNone(job)
        self.c.controller_event(job,'dispensing',OK)
        self.assertEqual(self.c.update_sensors(replace(OK,cup=False)).code,'no_cup')
        self.assertNotEqual(self.c.controller_event(job,'completed',OK).code,'completed')
        self.assertNotEqual(self.c.update_sensors(OK).code,'started')
        self.c.controller_event(None,'fault_cleared',OK)
        self.assertIsNone(self.c.turn('现在好了吗',plan('status'),OK).job_id)
    def test_trusted_completion_and_stale_events(self):
        a=self.c.turn(T,P,OK)
        self.assertNotEqual(self.c.controller_event('wrong','completed',OK).code,'completed')
        self.assertNotEqual(self.c.turn('茶泡好了吧',plan('chat'),OK).code,'completed')
        out=self.c.controller_event(a.job_id,'completed',OK)
        self.assertEqual(out.code,'completed')
        self.assertIn('完成',out.text)
        job=self.c.turn(T,P,OK).job_id
        self.assertNotEqual(job,a.job_id)
        self.assertEqual(self.c.controller_event(a.job_id,'completed',OK).code,'stale')
    def test_stop_busy(self):
        self.assertEqual(self.c.turn(T,P,OK).code,'started')
        self.assertEqual(self.c.turn(T,P,OK).code,'busy')
        self.assertEqual(self.c.turn('快停！',None,OK).code,'stopped')
    def test_stop_paraphrases(self):
        for text in ("别泡了", "先别出水了", "停止出水", "停机"):
            c=TeaController()
            self.assertEqual(c.turn(T,P,OK).code,"started")
            self.assertEqual(c.respond(text,plan("chat"),OK,self.no_chat)[0].code,"stopped")

    def test_user_declines_substitution(self):
        for p in (P,plan("clarify"),plan("chat")):
            r=self.c.turn("只接受恩施玉露，其他茶都别给我做。",p,OK)
            self.assertEqual(r.code,"no_substitution")
            self.assertIsNone(r.job_id)
            self.assertNotIn("？",r.text)

    def test_race(self):
        with ThreadPoolExecutor(5) as pool:
            results=list(pool.map(lambda _:self.c.turn(T,P,OK).code,range(5)))
        self.assertEqual(results.count('started'),1)
        self.assertEqual(results.count('busy'),4)
    def test_authoritative_error_and_invalid_code(self):
        for code in FAULTS:
            c=TeaController();job=c.turn(T,P,OK).job_id
            self.assertEqual(c.controller_event(job,code,OK).text,FAULTS[code])
            self.assertEqual(c.respond("明明泡好了吧？",plan('chat'),OK,self.no_chat)[1],FAULTS[code])
        self.assertEqual(self.c.controller_event(None,'other',OK).code,'sensor_unknown')
    def test_menu_from_config(self):
        c=TeaController(('龙井','白牡丹','铁观音'))
        out,_=c.respond('帮我泡杯茶',P,OK,self.no_chat)
        self.assertTrue(all(t in out.text for t in c.menu))
        self.assertNotIn('凤凰单丛',out.text)
        self.assertEqual(c.turn('给我泡杯白牡丹',plan('start','白牡丹','标准'),OK).code,'started')
    def test_duplicate_after_completion(self):
        j=self.c.turn(T,P,OK,sequence=20).job_id
        self.c.controller_event(j,'completed',OK)
        self.assertEqual(self.c.turn(T,P,OK,sequence=20).code,'duplicate')
        self.assertEqual(self.c.turn(T,P,OK,sequence=19).code,'duplicate')
        self.assertEqual(self.c.turn(T,P,OK,sequence=21).code,'started')
    def test_wrong_model_action_cannot_hide_missing_menu(self):
        for a in ('chat','clarify','status','stop'):
            x=TeaController().turn('请给我泡一杯茶',plan(a),OK)
            self.assertTrue(all(t in x.text for t in DEFAULT_MENU))
            self.assertIsNone(x.job_id)
    def test_model_cannot_cancel_in_negation(self):
        self.c.turn(T,P,OK)
        self.assertEqual(self.c.turn('继续加热，不要停止',plan('stop'),OK).code,'confirm_stop')
        self.assertIsNotNone(self.c.update_sensors(OK).job_id)
    def test_session_does_not_inherit_another_choice(self):
        self.c.turn("我选铁观音",plan('clarify'),OK,session="a")
        out=self.c.turn("确认开始",P,OK,session="b")
        self.assertNotEqual(out.code,"started")
        self.assertIsNone(out.job_id)
    def test_machine_menu_questions_bypass_model(self):
        for q in ("这台能泡龙井吗？","能不能做祁门红茶？","这台能泡什么茶？","泡龙井"):
            for proposed in (plan("chat"),P):
                out,answer=self.c.respond(q,proposed,OK,self.no_chat)
                self.assertIsNone(out.job_id)
                self.assertTrue(all(v in answer for v in DEFAULT_MENU))
        self.assertEqual(self.c.turn("给我泡一杯凤凰单丛",plan("start","凤凰单丛","标准"),OK).code,"started")

    def test_focused_reference_requires_new_authorization(self):
        for tea in DEFAULT_MENU:
            c=TeaController()
            p=plan("start",tea,"淡")
            r,words=c.respond(tea+"闻起来挺不错。",p,OK,self.no_chat,session="one")
            self.assertEqual(r.code,"chat")
            self.assertIsNone(r.job_id)
            self.assertIn(tea,words)
            self.assertNotIn(next(t for t in DEFAULT_MENU if t!=tea),words)
            self.assertEqual(c.turn("就这款，给我冲杯淡的，立即开泡。",p,OK,session="one").code,"started")

    def test_focus_cannot_override_wrong_tea_or_negation(self):
        for q,p in [("就这个，马上开泡。",plan("start","铁观音","标准")),
                    ("就这个，马上开泡。",plan("chat")),("就这个，别启动。",P),
                    ("我说过就这个，给我泡杯淡的，现在开始吧？",P)]:
            c=TeaController()
            c.turn("凤凰单丛听起来不错",plan("chat"),OK)
            self.assertIsNone(c.turn(q,p,OK).job_id)

    def test_focus_expires_and_does_not_cross_sessions(self):
        from unittest.mock import patch
        for delay,sess in [(91,"a"),(1,"b")]:
            c=TeaController()
            with patch("tea_controller.monotonic",return_value=10):
                c.turn("铁观音听起来不错",P,OK,session="a")
            with patch("tea_controller.monotonic",return_value=10+delay):
                self.assertIsNone(c.turn("就这个，马上开泡。",P,OK,session=sess).job_id)

    def test_menu_quote_and_intervening_text_are_not_a_focus(self):
        for first in ["室友说铁观音听起来不错","铁观音和凤凰单丛听起来不错","我不觉得铁观音听起来不错","铁观音听起来不错？"]:
            c=TeaController()
            c.turn(first,P,OK)
            self.assertIsNone(c.turn("就这个，马上开泡。",P,OK).job_id)
        c=TeaController()
        c.turn("铁观音听起来不错",P,OK)
        c.turn("先聊窗外的雨",plan("chat"),OK)
        self.assertIsNone(c.turn("就这个，马上开泡。",P,OK).job_id)

    def test_quote_and_sensor_fault(self):
        c=TeaController()
        r=c.turn("室友说‘开一杯铁观音’，我只是在转述。",P,OK)
        self.assertEqual(r.code,"chat")
        self.assertIsNone(r.job_id)
        c.turn("铁观音听起来不错",P,OK)
        self.assertEqual(c.turn("就这个，现在开泡。",P,Sensors(False,True,True,True)).code,"no_cup")

    def test_chat_and_status(self):
        self.assertEqual(self.c.respond("聊聊下雨天",plan('chat'),OK,lambda _:"听雨。")[1],"听雨。")
        self.assertEqual(self.c.respond('现在机器状态？',plan('chat'),OK,self.no_chat)[0].code,'status')

if __name__=='__main__':unittest.main()
