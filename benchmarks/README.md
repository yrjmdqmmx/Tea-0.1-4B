# 评测集与结果

suites/tea/ 提供项目自编的 60 道茶知识单选、30 道茶名辨识和 10 道开放题，含评判依据。preflight/device_cases.json 保存 18 轮设备题、8 段二轮对话的 16 轮以及 10 道额外指代题，合计 44 轮。10 道开放题与这 10 道额外设备题来自同一组 20 道留出题。它们出现在多个视图，统计时应使用对应分母。

Tea-Device-44.jsonl 是冻结 Q8_0 在同一控制器上的软件回放。可从仓库根目录复算：

```sh
python benchmarks/scripts/device_eval.py benchmarks/Tea-Device-44.jsonl --out device-scores.json
PYTHONPATH=exports/controller python -m unittest discover -s exports/controller/tests
PYTHONPATH=exports/controller python benchmarks/scripts/faults.py benchmarks/fault36.json /tmp/tea-faults.json
```

## 公开基准复现

五个外部套件共 838 题。public_subset_ids.json 给出固定题号，preflight/public_sources.json 列出上游版本、下载路径、文件摘要和各自许可。运行下面命令会从上游取回对应原文，种子为 42。请按各数据集自己的条款使用。

```sh
python3.11 -m venv .venv-bench
.venv-bench/bin/pip install -r benchmarks/preflight/public_requirements.txt
.venv-bench/bin/python benchmarks/scripts/public_build.py --cache .cache/public-bench --fetch
```

脚本会对冻结输入逐一校验。重建后的 JSONL 位于 benchmarks/suites/public/，脚本会对照 preflight/public_frozen.json 自动核对五个子集的原始哈希。AlignBench 仅选「综合问答」全部 38 题，C-Eval、Chinese-SimpleQA、IFEval-ZH、BFCL-v3 各 200 题。C-Eval 使用零样本。Chinese-SimpleQA v3 最终结果采用参考答案语义盲评，原始构建清单中的本地匹配记录描述较早版本。

## 已公开题目的解释

scores.json 与 comparison-3x.png 是 v3 历史记录。六款模型在 Colab G4 推理，Tea 用 Q8_0，其余五款用 bf16。客观项用贪婪解码，开放对话按冻结采样设置。软件硬件安全项按 44 轮中的错误事件统计。34/34 与 10/10 均计入确定性控制器。茶名辨识 16/30 是裸输出盲评的一项已知短板。公开后这些题可用于回归；新的独立泛化结论需要新留出题。
