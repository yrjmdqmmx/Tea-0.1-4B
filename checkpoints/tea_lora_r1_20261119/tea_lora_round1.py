"""Colab L4 首轮 LoRA；由 Notebook 传入已加载的官方模型和分词器。"""
from collections import Counter
from datetime import datetime
from hashlib import sha256
import importlib.metadata as md
import json
import math
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

SEED = 20261119
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_NAME = "Qwen/Qwen3.5-4B"
HASHES = {
    "train_180.jsonl": "e0adf1678a90e5c2c4a8783d75b121ec034e11a8c6a9d215c9e7d6463562b728",
    "val_20.jsonl": "afa3d4276a5a5893f6c7b532ed8592771f510ee0d2c7e2428e8a7253fd3af1d1",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def encode(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    end_marker = "<|im_end|>"
    spans, cursor = [], 0
    for msg in messages:
        if msg["role"] == "assistant":
            target = msg["content"].strip() + end_marker
            start = text.index(target, cursor)
            cursor = start + len(target)
            spans.append((start, cursor))
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    labels = [token if any(b > lo and a < hi for lo, hi in spans) else -100
              for token, (a, b) in zip(ids, enc["offset_mapping"])]
    assert spans and labels.count(tok.convert_tokens_to_ids(end_marker)) == len(spans)
    return dict(ids=ids, labels=labels, n=sum(x != -100 for x in labels[1:]))


def datasets(root, tok):
    folder = root / "data/tea_chat_sft_v0_2"
    split = json.loads((folder / "split_180_20.json").read_text())["outputs"]
    provenance = {r["line"]: r for r in map(json.loads, (folder/"provenance.jsonl").read_text().splitlines())
                  if r["split"] == "train"}
    result = []
    for name, expected in HASHES.items():
        data = (folder/name).read_bytes()
        assert sha256(data).hexdigest() == expected
        rows = []
        for line, src in zip(data.splitlines(), split[name]["source_lines"], strict=True):
            row = encode(tok, json.loads(line)["messages"])
            row["domain"] = provenance[src]["topic"][0]
            rows.append(row)
        assert Counter(x["domain"] for x in rows) == split[name]["domains"]
        result.append(rows)
    assert len(result[0]) == 180 and len(result[1]) == 20
    return result


def batch(rows, pad):
    width = max(len(x["ids"]) for x in rows)
    ids, labels, masks = [], [], []
    for item in rows:
        n = width-len(item["ids"])
        ids.append(item["ids"] + [pad]*n)
        masks.append([1]*len(item["ids"]) + [0]*n)
        labels.append(item["labels"] + [-100]*n)
    return {key: torch.tensor(val, device="cuda") for key, val in
            (("input_ids", ids), ("attention_mask", masks), ("labels", labels))}


@torch.inference_mode()
def score(model, rows, pad):
    model.eval()
    losses, counts = Counter(), Counter()
    for r in rows:
        value = model(**batch([r], pad), use_cache=False).loss.item()
        assert math.isfinite(value)
        losses[r["domain"]] += value * r["n"]
        counts[r["domain"]] += r["n"]
    return dict(nll=sum(losses.values()) / sum(counts.values()), target_tokens=sum(counts.values()),
                domain_nll={k: losses[k]/counts[k] for k in sorted(counts)})


def train(base, tok, project, run):
    # Colab 镜像的旧 torchao 会阻止 PEFT 注册普通线性层。仅更新可选库的官方 wheel。
    import subprocess, sys
    from packaging.version import Version
    if md.packages_distributions().get("torchao") and Version(md.version("torchao")) < Version("0.16"):
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                        "--only-binary=:all:", "torchao==0.18.0"], check=True)
    from huggingface_hub import hf_hub_download
    config_path = Path(hf_hub_download(MODEL_NAME, "config.json", revision=REV, local_files_only=True))
    assert config_path.parent.name == REV and base.config.model_type == "qwen3_5"
    assert "L4" in torch.cuda.get_device_name(0) and base.dtype == torch.bfloat16
    if (run/"training.json").exists() or (run/"adapter/adapter_model.safetensors").exists():
        raise FileExistsError(f"已有训练产物，保留在 {run}")
    train_rows, val = datasets(project, tok)
    metrics = run / "metrics.jsonl"
    def log(**record):
        record["time"] = now()
        with metrics.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(record, flush=True)

    torch.manual_seed(SEED)
    wanted = {"q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z",
              "out_proj", "gate_proj", "up_proj", "down_proj"}
    targets = [n for n, m in base.named_modules() if "language_model.layers." in n
               and n.split(".")[-1] in wanted and isinstance(m, torch.nn.Linear)]
    assert targets and any(n.endswith(".q_proj") for n in targets)
    config = dict(model=MODEL_NAME, revision=REV, hashes=HASHES, start=now(), seed=SEED, epochs=2,
                  r=8, alpha=16, dropout=0.05, batch=2, accumulation=4, peak_lr=1e-4,
                  warmup_steps=3, schedule="cosine", optimizer="AdamW", weight_decay=.01, clip=1.,
                  base_dtype=str(base.dtype), gpu=torch.cuda.get_device_name(0), enable_thinking=False,
                  target_counts=dict(Counter(x.split(".")[-1] for x in targets)),
                  versions={k: md.version(k) for k in ("torch","transformers","peft","accelerate")},
                  max_train_length=max(len(x["ids"]) for x in train_rows),
                  train_target_tokens=sum(x["n"] for x in train_rows), loss="assistant text and im_end only")
    write_json(run/"settings.json",config)
    pad = tok.pad_token_id
    original = score(base,val,pad)
    log(stage="original", **original)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias="none", target_modules=targets,
                     revision=REV, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    try:
        params = [p for n,p in model.named_parameters() if p.requires_grad]
        assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
        config["trainable_parameters"] = sum(p.numel() for p in params)
        config["adapter_dtypes"] = sorted({str(p.dtype) for p in params})
        write_json(run/"settings.json",config)
        total_steps = 2*math.ceil(len(train_rows)/8)
        opt = torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
        sched = get_cosine_schedule_with_warmup(opt,num_warmup_steps=3,num_training_steps=total_steps)
        step, best, epochs = 0, float("inf"), []
        for epoch in (1, 2):
            model.train()
            order = list(train_rows)
            random.Random(SEED+epoch).shuffle(order)
            loss_sum, n_sum = 0., 0
            for offset in range(0,len(order),8):
                block = order[offset:offset+8]
                count = sum(x["n"] for x in block)
                opt.zero_grad(set_to_none=True)
                total = 0.
                for k in range(0,len(block),2):
                    micro = block[k:k+2]
                    n = sum(x["n"] for x in micro)
                    loss = model(**batch(micro,pad),use_cache=False).loss
                    (loss*(n/count)).backward()
                    total += loss.detach().item()*n
                norm = torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                lr = opt.param_groups[0]["lr"]
                opt.step()
                sched.step()
                step += 1
                loss_sum += total
                n_sum += count
                log(stage="step", epoch=epoch, step=step, train_nll=total/count, grad_norm=norm.item(), lr=lr)
            valid = score(model,val,pad)
            epochs.append(dict(epoch=epoch, train_nll=loss_sum/n_sum, validation=valid))
            log(stage="epoch", **epochs[-1])
            if valid["nll"] < best:
                best, best_epoch = valid["nll"], epoch
                model.save_pretrained(run/"adapter",safe_serialization=True,save_embedding_layers=False)
        result = dict(start=config["start"], completed=now(), steps=step, original=original, epochs=epochs,
                      best_epoch=best_epoch, best_val_nll=best, peak_gpu_GB=torch.cuda.max_memory_allocated()/1e9,
                      adapter_sha256=sha256((run/"adapter/adapter_model.safetensors").read_bytes()).hexdigest())
        write_json(run/"training.json",result)
        print("TRAINING_COMPLETE",result,flush=True)
        return result
    finally:
        model.unload()
        base.eval()
