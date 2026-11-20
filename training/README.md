# 第五轮 SFT

从 Qwen/Qwen3.5-4B 官方对话版权重修订 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 开始。训练时保留原模型结构，对 32 个语言层中的 200 个线性投影挂 LoRA，可训练参数 30,474,240。纯文本聊天模板关闭思考，只有助手回答和结束标记参与 loss。每轮将 1,746 条对话按种子洗牌、各过一次。种子 20261123。

| 训练参数 | 值 |
|---|---|
| 精度与算子 | bf16，SDPA，纯 PyTorch 回退 |
| LoRA | r=16，训练 alpha=32，dropout=0.05 |
| 优化 | AdamW，学习率 4.5e-5，weight decay 0.01，梯度裁剪 1.0 |
| 调度与批量 | 16 步 warmup，cosine，有效批量 8，按长度调整微批 |
| 选模 | 比较第 219、274、329 步，选中 219；由 20 条旧开发对话、12 条语言和 12 条动作开发题综合选择 |
| 部署校准 | 只在开发数据上比较 alpha 32、24、28、36，最终为 28（缩放 28/16） |

训练合计 329 步。最终采用的第 219 步对应第一个完整 epoch。DPO 实验单独归档，发布权重仅含 SFT。

## 复现

在全新 NVIDIA L4 环境克隆本仓库。原训练环境使用 Python 3.13.16、torch 2.14.1+cu130、Transformers 5.23.0、PEFT 0.21.3、Accelerate 1.16.0。匹配 CUDA 的 PyTorch 应由运行环境提供。

```sh
python -m pip install transformers==5.23.0 peft==0.21.3 accelerate==1.16.0 safetensors==0.8.0
python training/run.py
```

结果保存在 runs/round5。原始训练和校准记录在 records/。train_merged.jsonl 的 SHA-256 和原始基座损失均由训练脚本检查。浮点计算和驱动版本可能影响复现出的权重摘要。GitHub Release 中另附选中步的原始 LoRA 以及部署配置。

## 导出

以 bf16 加载同一基座，PEFT 加载部署配置后用 merge_and_unload(safe_merge=True) 合并。对 200 个目标投影各取一个受影响切片核对更新前后数值，并跑茶知识、启动 JSON、非思考三条冒烟。原始核对记录在 records/merge_smoke.json。使用 llama.cpp b12532（08c252e221759e0f69e80f0450b295c92d2344e1）走纯文本转换、禁用 MTP，保留独立投影张量名，再由同一 bf16 GGUF 量化为 Q8_0。records/conversion.json 记录了转换选项和导出哈希。附带的 Q4_K_M 历史记录仅用于说明量化差异；本次冻结发布选择 Q8_0。
