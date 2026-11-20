# Tea-0.1-4B

Tea-0.1-4B 用中文聊茶、接日常闲聊，也能把“刚才那款，给我泡淡一点”转换为泡茶机的动作提议。模型根据历史对话提出茶种和浓度，控制器核对菜单、用户授权、传感器与任务回执后再决定是否执行。当前原型菜单为铁观音和凤凰单丛。

[下载完整 Q8_0](https://huggingface.co/yrjmdqmmx/Tea-0.1-4B/resolve/main/qwen35-4b-tea-v5-Q8_0.gguf) · [GitHub 权重分卷与 LoRA](https://github.com/yrjmdqmmx/Tea-0.1-4B/releases/tag/v0.1.0) · [公开训练与评测数据](https://huggingface.co/datasets/yrjmdqmmx/Tea-0.1-4B-Data) · [训练方法与记录](https://github.com/yrjmdqmmx/Tea-0.1-4B/tree/main/training)

## 输入、输出与上下文

| 项目 | 发布版本 |
|---|---|
| 输入 | 文本；控制器可附上本轮设备状态和对话历史 |
| 输出 | 中文文本，或供控制器解析的动作 JSON |
| 推理模板 | 非思考模式；部署预设为贪婪解码 |
| 本地默认窗口 | 4,096 tokens，含系统提示、历史、当前输入与生成；单次生成默认上限 192 tokens |
| 比较评测 | 每个并发请求预留 32,768 tokens；未开展该长度下的长文检索、长对话记忆验收 |
| 视觉与声音 | 本 GGUF 仅含语言模型。图片、视频及音频输入均未交付。语音交互需要独立的语音识别和合成模块 |

官方基座带视觉编码器，支持文本、图像、视频输入和文本输出，标称原生窗口 262,144 tokens。Tea 的专用微调、量化和设备联调均走文本路径。基座的窗口上限也未在 Tea 上做专项验证。[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)

## 本地运行

克隆工程，下载 GGUF 并检查摘要。在 macOS 或 Linux 上安装 Ollama、curl 和 Python 3.10+，然后运行：

```sh
git clone https://github.com/yrjmdqmmx/Tea-0.1-4B.git
cd Tea-0.1-4B/exports
mkdir -p gguf
curl -fL --retry 3 -o gguf/qwen35-4b-tea-v5-Q8_0.gguf https://huggingface.co/yrjmdqmmx/Tea-0.1-4B/resolve/main/qwen35-4b-tea-v5-Q8_0.gguf
shasum -a 256 -c SHA256SUMS
sh run_local.sh ollama
```

Linux 也可用 `sha256sum -c SHA256SUMS`。使用 llama.cpp，可改用 `sh run_local.sh llama`。脚本使用模拟设备。物理加热、阀门、干烧保护仍需硬件联锁和实机验证。仅聊天可在 Ollama 导入后运行 `ollama run tea-v5`。

通过 GitHub 下载时，在本仓库的 exports 目录运行下面的命令。需要 GitHub CLI；已手动下载资产时可跳过第一行。

```sh
gh release download v0.1.0 --repo yrjmdqmmx/Tea-0.1-4B --pattern '*.part0?'
shasum -a 256 -c SHA256SUMS.parts
mkdir -p gguf
cat qwen35-4b-tea-v5-Q8_0.gguf.part0[123] > gguf/qwen35-4b-tea-v5-Q8_0.gguf
shasum -a 256 -c SHA256SUMS
```

整文件 4,482,402,816 字节，SHA-256 为 `15735c5ddcf019338cf9ff78bd28fc313524a9290b6b09e335eda1e5b7f1bc59`。

## 训练与评测

第五轮 SFT 从固定修订的 Qwen3.5-4B 对话版重新训练。1,746 段对话覆盖茶知识、自然聊天、动作提议和按控制器回执播报。LoRA 秩 16，训练 alpha 32，部署 alpha 28。使用开发集选中的第 219 步，核对 200 个目标投影后合并与量化。历史 DPO 实验未纳入成品。[原始训练与校准记录](https://github.com/yrjmdqmmx/Tea-0.1-4B/tree/main/training/records)

| v3 固定样本 | 得分 |
|---|---:|
| 固定软件闭环 / 额外多轮指代 | 34/34 · 10/10 |
| 软件控制器误播 / 漏启 / 误启 / 指错茶 | 0 / 0 / 0 / 0 |
| 茶知识单选 / 陌生茶名防编造 / 茶文化开放题 | 50/60 · 16/30 · 7/10 |
| C-Eval / Chinese-SimpleQA | 150/200 · 107/200 |

茶名辨识仍有 14 道未通过；购茶时不能仅凭商品名判断品种、产区或真伪。设备通过率包含确定性控制器。评分来自本项目的固定子集。[12 项得分与评测方法](https://github.com/yrjmdqmmx/Tea-0.1-4B/tree/main/benchmarks)

## 来源与许可

基座为 Qwen 团队的 Qwen/Qwen3.5-4B（Apache-2.0），本项目进行了中文 SFT、权重合并与 Q8_0 量化。原创控制器与代码采用 Apache-2.0，原创训练与评测数据采用 CC BY 4.0，第三方素材的逐项许可见发布清单。[许可及署名](https://github.com/yrjmdqmmx/Tea-0.1-4B/blob/main/NOTICE.md)
