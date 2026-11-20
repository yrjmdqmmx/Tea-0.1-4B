"""用户指代、会话隔离和启动授权。测试菜单可替换，模型字段始终视为提议。"""
import unittest
from tea_controller import TeaController, Sensors

OK = Sensors(True, True, True, True)
CHAT = dict(action="chat", tea=None, strength=None)


def start(tea, strength="标准"):
    return dict(action="start", tea=tea, strength=strength)


class SelectionHistoryTests(unittest.TestCase):
    def test_reserve_without_start_even_if_model_orders(self):
        for tea in ("铁观音", "凤凰单丛"):
            for wording in ("替我预选{}", "{}先记着", "我先把{}圈起来", "先留着{}"):
                with self.subTest(tea=tea, wording=wording):
                    ctl = TeaController()
                    saved = ctl.turn(wording.format(tea), start(tea), OK)
                    self.assertEqual((saved.code, saved.job_id), ("selected", None))
                    self.assertEqual(ctl.turn("按之前选的马上冲一杯", start(tea), OK).code, "started")

    def test_explicit_registered_reference_survives_social_turn(self):
        ctl = TeaController()
        ctl.turn("我先登记凤凰单丛，淡档", CHAT, OK)
        ctl.turn("周末的电影挺好看", CHAT, OK)
        self.assertEqual(ctl.turn("帮我泡登记的那款", start("凤凰单丛", "淡"), OK).code, "started")

    def test_own_candidate_can_be_registered(self):
        ctl = TeaController()
        ctl.turn("铁观音闻起来不错", CHAT, OK)
        self.assertEqual(ctl.turn("帮我记住这款", start("铁观音"), OK).code, "selected")
        self.assertEqual(ctl.turn("就它，立即启动", start("铁观音"), OK).code, "started")

    def test_switch_is_a_selection_until_requested(self):
        c = TeaController()
        c.turn("我先暂定铁观音", CHAT, OK)
        self.assertEqual(c.turn("换成另一种", start("凤凰单丛"), OK).code, "selected")
        self.assertEqual(c.turn("刚才选的马上开始", start("凤凰单丛"), OK).code, "started")

    def test_multi_tea_menu_elimination_is_unique(self):
        c = TeaController(("甲茶", "乙茶", "丙茶"))
        c.turn("先排除乙茶", CHAT, OK)
        self.assertIsNone(c.turn("做剩下的", start("甲茶"), OK).job_id)
        c.turn("甲茶也不要", CHAT, OK)
        self.assertEqual(c.turn("给我泡剩余的那款", start("丙茶"), OK).code, "started")

    def test_revoked_choice_stays_revoked_after_ambiguous_order(self):
        c = TeaController(("甲茶", "乙茶", "丙茶"))
        c.turn("先选乙茶", CHAT, OK)
        r = c.turn("去掉登记的，做剩下的", start("丙茶"), OK)
        self.assertEqual((r.code, r.job_id), ("choose_tea", None))
        self.assertIsNone(c.turn("泡刚才选的", start("乙茶"), OK).job_id)
        c.turn("排除甲茶", CHAT, OK)
        self.assertEqual(c.turn("做剩余那种", start("丙茶"), OK).code, "started")

    def test_wrong_model_fields_are_blocked_in_both_directions(self):
        for tea, other in (("甲茶", "乙茶"), ("乙茶", "甲茶")):
            for proposal in (start(tea), start(other, "淡"), CHAT, '{"action":"start"}'):
                c = TeaController(("甲茶", "乙茶"))
                c.turn("先把"+tea+"记下", CHAT, OK)
                r = c.turn("去掉登记的，立即做另一款", proposal, OK)
                self.assertIsNone(r.job_id, proposal)

    def test_only_excluded_item_can_never_start_by_reference(self):
        c = TeaController(("甲茶", "乙茶"))
        c.turn("登记甲茶", CHAT, OK)
        c.turn("划去刚记下的", CHAT, OK)
        self.assertIsNone(c.turn("按刚登记的泡一杯", start("甲茶"), OK).job_id)
        self.assertEqual(c.turn("请做剩下的一种", start("乙茶"), OK).code, "started")

    def test_no_decision_never_inherits_model_or_another_guest(self):
        for intro in ("上一位要铁观音", "两款都挺香", "朋友说先登记凤凰单丛", "铁观音和凤凰单丛"):
            for order in ("来杯淡茶", "请做另一款", "按之前选的启动"):
                c = TeaController()
                c.turn(intro, start("凤凰单丛"), OK)
                r = c.turn(order, start("凤凰单丛", "淡"), OK)
                self.assertIsNone(r.job_id)
                self.assertTrue(all(t in r.text for t in c.menu))

    def test_missing_choice_gets_full_menu_even_for_chat_proposal(self):
        for text in ("确认开始", "现在启动", "就这款，马上开泡", "泡杯淡茶"):
            c = TeaController()
            r = c.turn(text, CHAT, OK)
            self.assertEqual((r.code,r.job_id), ("choose_tea", None), text)
            self.assertTrue(all(t in r.text for t in c.menu), text)

    def test_switch_guest_and_cancel_invalidate_saved_tea(self):
        for reset in ("我先留着白牡丹", "我自己还没选，能调淡吗？", "登记先作废", "这杯先取消"):
            c = TeaController()
            c.turn("暂存铁观音", CHAT, OK)
            c.turn(reset, CHAT, OK)
            self.assertIsNone(c.turn("请按登记的马上启动", start("铁观音"), OK).job_id)
        c = TeaController()
        c.turn("暂定铁观音", CHAT, OK, session="甲")
        self.assertIsNone(c.turn("泡刚才选的", start("铁观音"), OK, session="乙").job_id)

    def test_no_authority_in_questions_deferrals_quotes_or_conditions(self):
        for text in ("要不要登记铁观音？", "假如预选铁观音", "他让我登记铁观音",
                     "圈定铁观音，或者凤凰单丛", "预选铁观音，预选凤凰单丛"):
            c = TeaController()
            c.turn(text, start("铁观音"), OK)
            self.assertIsNone(c.turn("做刚登记的", start("铁观音"), OK).job_id)
        for text in ("等客人到了做登记的", "做登记的，先别启动", "泡刚选的？", "开另一款，还是再等等"):
            c = TeaController()
            c.turn("预选凤凰单丛", CHAT, OK)
            self.assertIsNone(c.turn(text, start("凤凰单丛"), OK).job_id)

    def test_model_wrong_or_busy_never_creates_second_job(self):
        c = TeaController()
        c.turn("预选凤凰单丛", CHAT, OK)
        self.assertIsNone(c.turn("做选的那一种", start("铁观音"), OK).job_id)
        r = c.turn("做登记的", start("凤凰单丛"), OK)
        self.assertEqual(r.code,"started")
        again = c.turn("请泡一杯铁观音", start("铁观音"), OK)
        self.assertEqual((again.code, again.job_id), ("busy", r.job_id))

    def test_fault_and_stop_clear_exclusions_and_registrations(self):
        for interrupt in ("fault", "stop"):
            c = TeaController()
            c.turn("去掉凤凰单丛", CHAT, OK)
            if interrupt == "fault":
                c.update_sensors(Sensors(False, True, True, True))
                c.controller_event(None, "fault_cleared", OK)
            else:
                c.turn("停一下", CHAT, OK)
            self.assertIsNone(c.turn("启动剩下的", start("铁观音"), OK).job_id)
