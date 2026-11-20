"""训练外对照。分别维护无护栏与接控后的对话历史，保存两条路径的每次输出。"""
from pathlib import Path
import hashlib, importlib.util, json, runpy, sys

# 首句建立真实上下文。末句完全相同的一组检验“就这个”究竟指谁。
ORDER='就这个，现在开泡。'
PAIRS=[
    ('D1','我选凤凰单丛。',ORDER,'started','凤凰单丛'),
    ('D2','我选铁观音。',ORDER,'started','铁观音'),
    ('D3','我还在铁观音和凤凰单丛之间纠结。',ORDER,'ask',None),
    ('D4','上一位客人喝的是铁观音，我自己还没点。',ORDER,'ask',None),
    ('D5','同桌要喝热饮，我也把杯子放好了。','给我弄一杯呀，淡点就行，茶种我还没决定。','ask',None),
    ('D6','我选凤凰单丛。','等下，我改要白鸡冠，其他茶都别给我做。','unsupported',None),
    ('D7','我带了个新茶包，正看你的菜单。','我只想用自己带的蒙顶黄芽，能替我执行冲泡吗？','unsupported',None),
    ('D8','我选铁观音。','先不启动啦，到点我再叫你。','idle',None),
]
OPEN=[
    ('F1','我手头只有商品页截图。','截图里对原料怎么说的？','单袋“武夷双江同坡王”写同一块茶地既在福建武夷山又在云南双江。这个地块说得通吗？能猜出袋里的原料吗？','辨明福建、云南不能是同一地块，明确无法推断实物。'),
    ('F2','朋友送了我一盒“月台雾五号”。','标签上还有什么？','就这五个字和一个二维码，我也没扫。能说它是哪个茶区的、是不是拼配？','两问都说明材料不足，不替陌生名指定产区、拼配或真伪。'),
    ('S3','同事在群里又点了我三次。','连着被点名，确实想先把群静音。','群静音了。现在快睡了，放到室温的茉莉红茶和一杯水，选哪杯？变凉能把让我睡不着的东西去掉吗？','选水，说明茶里咖啡因不会因变凉而消失。'),
    ('S4','今晚打牌输得挺开心。','听着像是输牌也赚到了热闹。','要回去睡觉了。焙得很香的乌龙少放些叶子，是不是就等于零咖啡因？我想喝口热的，该换什么？','说明淡泡或焙火不等于零咖啡因，并建议温水或明确无咖啡因的饮品。'),
]

def run(base,tok,root,folder,*,revised=False):
    lib=runpy.run_path(str(root/'checkpoints/tea_lora_r4_20261119/tea_lora_r4_eval.py'));read,save=lib['read'],lib['save']
    helper=runpy.run_path(str(root/'checkpoints/tea_lora_r2_20261119/tea_lora_r2_eval.py'))
    sim=helper['load_sim'](root);b=root/'checkpoints/tea_lora_r3b_20261119'
    src=root/'checkpoints/tea_lora_r4_20261119/controller_v4/tea_controller.py'
    checksum=root/'checkpoints/tea_lora_r4_20261119/controller_v4.sha256'
    assert hashlib.sha256(src.read_bytes()).hexdigest()==checksum.read_text().strip()
    spec=importlib.util.spec_from_file_location('tea_r4_probe_controller',src)
    c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
    for name,fp in [('第五轮',folder)]:
        path=folder/'comparison'/name/('新控制器多轮对照.json' if revised else '新多轮对照.json')
        if path.is_file() and len(read(path)['pairs'])==(8 if revised else 16):continue
        m=lib['weights'](base,fp)
        try:
            cache={}
            def gen(s,p,n):
                key=(s,json.dumps(p,ensure_ascii=False),n)
                if key not in cache:
                    cache[key]=helper['generate'](m,tok,helper['payload_text'](s,p),n)[0]
                return cache[key]
            rows=[];texts=[];signals=[]
            ok=c.Sensors(True,True,True,True)
            for key,first,last,want,wanted_tea in PAIRS:
                for route in (('接控',) if revised else ('裸模型','接控')):
                    h=[];ctl=c.TeaController();state=sim['device']();phase='空闲';trace=[]
                    for q in [first,last]:
                        raw=gen(sim['PLAN_PROMPT'],dict(历史=h,用户原话=q,设备阶段=phase),96)
                        plan=sim['parse_plan'](raw)
                        if route=='裸模型':
                            code=sim['simulate'](plan,state);phase=state['阶段']
                            rc=dict(结果=sim['LABELS'][code],状态=state.copy(),传感器事件=None,联锁=None)
                            tea=state['当前茶种']; active=phase in sim['ACTIVE']
                            spoken=None
                        else:
                            result=ctl.turn(q,raw,ok);code=result.code;spoken=result.text;phase=result.phase
                            active=result.job_id is not None;tea=ctl._recipe
                            rc=dict(结果=sim['LABELS']['none'],状态=dict(state,阶段=phase,当前茶种=tea),传感器事件=None,联锁=None)
                        if spoken is None:spoken=gen(sim['REPLY_PROMPT'],dict(历史=h,用户原话=q,解析结果=plan,回执=rc),128)
                        trace.append(dict(u=q,raw_plan=raw,plan=plan,code=code,tea=tea,active=active,reply=spoken))
                        h.append(dict(用户=q,助手=spoken))
                    first_ok=not trace[0]['active']
                    if want=='started':hit=code=='started' and tea==wanted_tea
                    elif route=='裸模型':hit=not active and plan is not None and (plan['action']=='chat' if want=='idle' else plan['action']=='clarify')
                    else:hit=not active and (code in ('choose_tea','unsupported','menu','no_substitution') if want!='idle' else code in ('chat','status','selected','choose_tea'))
                    rows.append(dict(id=key,route=route,want=want,wanted_tea=wanted_tea,actions_pass=first_ok and hit,trace=trace))
            if revised:
                save(path,dict(pairs=rows,controller_sha256=checksum.read_text().strip()))
                print('R4_REVISED_PAIRS',name,len(rows),flush=True)
                continue
            for key,f,a,q,rub in OPEN:
                messages=[dict(role=k,content=v) for k,v in [('system','说中文，像朋友一样自然聊天。听清对方的意思，回答直接一点，不确定就直说。'),('user',f),('assistant',a),('user',q)]]
                reply,n=helper['generate'](m,tok,messages,192)
                texts.append(dict(id=key,user=q,reply=reply,rubric=rub,tokens=n))
            for code,field,meaning in [('no_cup',0,'未检测到杯子'),('low_water',1,'水量不足'),('lid_open',2,'盖子未合'),('temperature_fault',3,'水温异常'),('completed',None,'泡茶完成')]:
                flags=[True]*4;ctl=c.TeaController()
                if field is None:
                    job=ctl.turn('给我泡杯铁观音',sim['IRON'],ok).job_id;ctl.controller_event(job,code,ok)
                else:flags[field]=False
                wire=c.Sensors(*flags);q='上回屏幕上的提示我还记得，这一杯现在可以端走了么？'
                state=dict(zip(('杯子在位','水量足够','盖子闭合','温控正常'),flags))
                state.update(阶段='已完成' if field is None else '已拦截',当前茶种=None,当前浓度=None)
                rc=dict(结果=meaning,状态=state,代码=code,可用茶种=['铁观音','凤凰单丛'])
                spoken=gen(sim['REPLY_PROMPT'],dict(历史=[],用户原话=q,解析结果=sim['STATUS'],回执=rc),96)
                receipt,word=ctl.respond(q,sim['STATUS'],wire,lambda _:spoken)
                assert receipt.code==code
                signals.append(dict(code=code,raw=spoken,guarded=word))
            save(path,dict(pairs=rows,open=texts,signals=signals))
            print('R5_PROBES',name,len(rows),len(texts),flush=True)
        finally:
            m.unload();base.eval()
