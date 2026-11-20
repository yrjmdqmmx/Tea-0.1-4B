"""以本机推理服务驱动冻结的软件控制器。命令行使用模拟传感器。"""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen
from tea_controller import Sensors, TeaController

PROMPTS=json.loads(Path(__file__).with_name('prompts.json').read_text(encoding='utf-8'))

def render_text(messages):
    # 冻结控制器仅生成 system/user 的纯文本消息；支持普通多轮文本历史。
    history=''.join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in messages)
    return history+'<|im_start|>assistant\n<think>\n\n</think>\n\n'

def create_infer(backend, endpoint, model):
    def infer(messages,limit):
        if backend=='ollama':
            route='/api/chat'
            body=dict(model=model,messages=messages,think=False,stream=False,options=dict(num_predict=limit,temperature=0,top_k=1,repeat_penalty=1,seed=21928))
        else:
            route='/completion'
            body=dict(prompt=render_text(messages),n_predict=limit,temperature=0,top_k=1,repeat_penalty=1,cache_prompt=False,stream=False,seed=21928)
        req=Request(endpoint.rstrip('/')+route,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=180) as f:reply=json.load(f)
        return (reply['message']['content'] if backend=='ollama' else reply['content']).strip()
    return infer

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=['llama','ollama'],default='llama')
    ap.add_argument('--endpoint')
    ap.add_argument('--model',default='tea-v5')
    ap.add_argument('--simulate',action='store_true',required=True,help='使用软件模拟，便于在开发板上联调')
    args=ap.parse_args()
    endpoint=args.endpoint or ('http://127.0.0.1:11434' if args.backend=='ollama' else 'http://127.0.0.1:8080')
    infer=create_infer(args.backend,endpoint,args.model)
    controller=TeaController();history=[]
    sensors=Sensors(True,True,True,True)
    print('软件模拟 · 输入 /quit 退出。')
    while True:
        try:user=input('你 > ').strip()
        except (EOFError,KeyboardInterrupt):break
        if user=='/quit':break
        if not user:continue
        receipt,answer,_=controller.run_turn(user,history,infer,lambda:sensors,plan_prompt=PROMPTS['plan'],reply_prompt=PROMPTS['reply'])
        history.append(dict(用户=user,助手=answer))
        print(f'茶机 > {answer}\n[{receipt.code} · {receipt.phase} · {receipt.job_id or "无任务"}]',flush=True)
if __name__=='__main__':main()
