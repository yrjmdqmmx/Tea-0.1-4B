"""Reproduce round 5 on an L4. Run from any directory with python training/run.py."""
import runpy
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
BASE = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA L4":
        raise RuntimeError("The original training recipe requires an NVIDIA L4.")
    run = ROOT / "runs" / "round5"
    run.mkdir(exist_ok=True, parents=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REV)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE, revision=REV, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", use_kernels=False, use_safetensors=True,
    ).eval()
    runpy.run_path(str(ROOT / "training/tea_lora_round5.py"))["train"](model, tok, ROOT, run)
    runpy.run_path(str(ROOT / "training/tea_lora_r5_calibrate.py"))["calibrate"](model, tok, ROOT, run)


if __name__ == "__main__":
    main()
