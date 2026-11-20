#!/bin/sh
# 启动本机推理服务和软件控制器；退出时回收本脚本启动的服务。
set -eu
cd "$(dirname "$0")"
backend=${1:-ollama}
owned_pid=
cleanup() {
    if [ -n "$owned_pid" ]; then kill "$owned_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_service() {
    for attempt in $(seq 1 90); do
        if curl -fsS "$1" >/dev/null 2>&1; then return; fi
        if [ -n "$owned_pid" ] && ! kill -0 "$owned_pid" 2>/dev/null; then break; fi
        sleep 1
    done
    echo "服务未就绪。请查看上方服务日志。" >&2
    exit 1
}
case "$backend" in
ollama)
    port=${TEA_PORT:-11434}
    endpoint="http://127.0.0.1:$port"
    export OLLAMA_HOST="127.0.0.1:$port"
    if ! curl -fsS "$endpoint/api/version" >/dev/null 2>&1; then
        ollama serve >&2 &
        owned_pid=$!
        wait_service "$endpoint/api/version"
    fi
    ollama create tea-v5 -f Modelfile
    python3 -B controller/chat_controller.py --simulate --backend ollama --endpoint "$endpoint"
    ;;
llama)
    port=${TEA_PORT:-8080}
    endpoint="http://127.0.0.1:$port"
    if curl -fsS "$endpoint/health" >/dev/null 2>&1; then
        echo "该端口已有服务。请设置一个空闲的 TEA_PORT，或按 README 连接已知模型。" >&2
        exit 1
    fi
    "${LLAMA_SERVER:-llama-server}" -m "${TEA_GGUF:-gguf/qwen35-4b-tea-v5-Q8_0.gguf}" -c 4096 -np 1 -ngl "${TEA_GPU_LAYERS:-99}" --no-cache-prompt --host 127.0.0.1 --port "$port" >&2 &
    owned_pid=$!
    wait_service "$endpoint/health"
    python3 -B controller/chat_controller.py --simulate --backend llama --endpoint "$endpoint"
    ;;
*) echo "用法：sh run_local.sh [ollama|llama]" >&2; exit 2 ;;
esac
