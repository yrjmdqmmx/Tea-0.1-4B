from dataclasses import asdict
from unittest.mock import patch
import uuid

def held_device(model,tok,suite,helper,sim,c):
    data=[]
    for i,(key,first,ack,last,act,tea,strength) in enumerate(suite['device']):
        ids=iter(uuid.uuid5(uuid.NAMESPACE_OID,f'step5a-{key}-{j}') for j in range(12))
        with patch.object(c,'uuid4',side_effect=lambda:next(ids)):
            ctl=c.TeaController()
            sensors=c.Sensors(True,True,True,True)
            def run(u,h):
                audible=[]
                rc,text,trace=ctl.run_turn(u,h,lambda ms,lim:helper['generate'](model,tok,ms,lim)[0],lambda:sensors,
                          plan_prompt=sim['PLAN_PROMPT'],reply_prompt=sim['REPLY_PROMPT'],emit=audible.append)
                assert text==''.join(audible)
                assert trace['plan_input'][0]['content'].splitlines()[-1].startswith('[实时状态:')
                return dict(receipt=asdict(rc),spoken=text,recipe=ctl._recipe,strength=ctl._strength,**trace)
            intro=run(first,[])
            final=run(last,[dict(用户=first,助手=ack)])
            code=final['receipt']['code']
            if act=='start':
                passed=code=='started' and final['recipe']==tea and final['strength']==strength
            elif act=='clarify':
                passed=code in {'choose_tea','unsupported','no_substitution'} and not final['receipt']['job_id']
            elif key=='P10':
                passed=code=='capability' and '支持淡档' in final['spoken'] and not final['receipt']['job_id']
            else:
                passed=code in {'selected','chat','status'} and not final['receipt']['job_id']
            passed=bool(passed and not intro['receipt']['job_id'])
            data.append(dict(id=key,expected=sim['decision'](act,tea,strength),fixed_history_reply=ack,intro=intro,final=final,passed=passed))
    return data
