from dataclasses import asdict,replace
from pathlib import Path
from tea_controller import Sensors, TeaController
import json,sys

def replay(fixtures):
    chat = dict(action='chat',tea=None,strength=None)
    proposal = dict(action='start',tea='铁观音',strength='标准')
    signals = {'缺杯':'cup','缺水':'water','盖未合':'lid','温控异常':'temperature_ok'}
    cases = []
    for row in fixtures:
        phase = row['state']
        ctl=TeaController()
        wire=Sensors(True, True, True, True)
        if phase != '空闲':
            job = ctl.turn('请泡一杯铁观音',proposal,wire).active_job_id
            assert job
            if phase in ('出水中','已完成'):
                ctl.controller_event(job,'dispensing' if phase=='出水中' else 'completed',wire)
            elif phase == '已停止':
                ctl.turn('先停',chat,wire)
            elif phase in signals:
                wire=replace(wire,**{signals[phase]:False})
                ctl.update_sensors(wire)
            else:
                assert phase == '加热中'
        wanted=ctl.update_sensors(wire)
        receipt,spoken=ctl.respond("聊聊下雨天",chat,wire,lambda _:row['model_draft'])
        old=row['receipt']
        same_state=(wanted.phase,wanted.code,wanted.text,bool(wanted.job_id)) == (old['phase'],old['code'],old['text'],bool(old['job_id']))
        ok=same_state and receipt.job_id==wanted.job_id and spoken==row['final'] and spoken!=row['model_draft']
        cases.append(dict(state=phase,model_draft=row['model_draft'],receipt=asdict(receipt),spoken=spoken,passed=ok))
    assert len({(x['state'],x['model_draft']) for x in cases})==len(cases)==36
    assert all(x['passed'] for x in cases)
    return cases
if __name__=='__main__':
    fixtures=json.loads(Path(sys.argv[1]).read_text())['injected']
    cases=replay(fixtures)
    Path(sys.argv[2]).write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    print('Fault matrix: 36/36')
