"""Rebuild the frozen official-data subsets from pinned, SHA-256-checked files."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
from urllib.request import urlopen

BASE = Path(__file__).resolve().parents[1]
PREFLIGHT = BASE / "preflight"
DEST = BASE / "suites/public"
SEED = 42
EXPECTED = {"AlignBench-Chat": 38, "C-Eval": 200, "Chinese-SimpleQA": 200,
            "IFEval-ZH": 200, "BFCL-v3": 200}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_sources(cache, fetch):
    manifest = json.loads((PREFLIGHT / "public_sources.json").read_text())
    for entry in manifest["inputs"]:
        path = cache / entry["path"]
        if fetch and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(entry["url"], timeout=120) as response:
                data = response.read()
            if digest(data) != entry["sha256"]:
                raise ValueError(f"Download hash mismatch: {entry['path']}")
            path.write_bytes(data)
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Source hash mismatch: {entry['path']}")


def record(suite, key, prompt, messages=None, **ref):
    return dict(id=f"{suite}/{key}", suite=suite,
                messages=messages or [{"role": "user", "content": prompt}],
                ref={"source_prompt": prompt, **ref})


def frozen_sample(rows, count, rng):
    ordered = sorted(rows, key=lambda r: r["id"])
    return rng.sample(ordered, count)


def date_or_title_aliases(answer):
    """Freeze only predefined equivalent formats before reading any model output."""
    answer = answer.strip()
    aliases = set()
    date = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})[日号]", answer)
    if date:
        year, month, day = (int(v) for v in date.groups())
        aliases.update([f"{year:04}-{month:02}-{day:02}", f"{year}/{month}/{day}",
                        f"{year}年{month}月{day}日"])
    if re.fullmatch(r"\d{4}年", answer):
        aliases.add(answer[:-1])
    if re.fullmatch("《[^《》]+》", answer):
        aliases.add(answer[1:-1])
    # Accept an alternate spelling only with a source-supplied explicit alias marker.
    alias = re.fullmatch(r"([^()（）]+)[（(](?:又名|亦称|又称|别称为|英文名为)[：:]?([^,，;；()（）]+)[)）]", answer)
    if alias:
        aliases.update(s.strip() for s in alias.groups())
    return sorted(aliases.difference({answer, ""}))


def build(cache):
    from evalscope.api.registry import get_benchmark
    from evalscope.benchmarks import load_benchmark
    from evalscope.benchmarks.ceval.ceval_adapter import SUBJECT_MAPPING
    from evalscope.utils.io_utils import jsonl_to_list, parquet_to_list
    from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_FORMAT
    from bfcl_eval.model_handler.utils import formulate_system_prompt

    load_benchmark("ceval")
    load_benchmark("chinese_simpleqa")
    ceval = get_benchmark("ceval")
    simple = get_benchmark("chinese_simpleqa")
    out = {}
    align = [record("AlignBench-Chat", r["question_id"], r["question"],
                    source_id=r["question_id"], category=r["category"], subcategory=r["subcategory"],
                    answer=r["reference"], evidences=r["evidences"])
             for r in jsonl_to_list(str(cache / "align/data/data_v1.1_release.jsonl"))
             if r["category"] == "综合问答"]
    out["AlignBench-Chat"] = frozen_sample(align, len(align), random.Random(SEED))

    # Allocate 200 validation cases proportionally by subject using largest remainders.
    files = sorted((cache / "ceval").glob("*/val-00000-of-00001.parquet"))
    pools = {f.parent.name: parquet_to_list(str(f)) for f in files}
    assert set(pools) == set(SUBJECT_MAPPING)
    total = sum(map(len, pools.values()))
    quotas = {s: len(pool)*200//total for s, pool in pools.items()}
    by_remainder = sorted(pools, key=lambda s: (-(len(pools[s])*200 % total), s))
    for s in by_remainder[:200-sum(quotas.values())]:
        quotas[s] += 1
    assert min(quotas.values()) >= 1
    rng = random.Random(SEED)
    choices = []
    for subject, pool in pools.items():
        ceval.current_subset_name = subject
        for r in rng.sample(sorted(pool, key=lambda x: x["id"]), quotas[subject]):
            sample = ceval.record_to_sample(r)
            prompt = ceval.format_prompt_template(sample)
            choices.append(record("C-Eval", f"{subject}/{r['id']}", sample.input,
                                  messages=[dict(role="user",content=prompt)], subject=subject,
                                  category=SUBJECT_MAPPING[subject][2], answer=sample.target,
                                  source_id=r["id"], split="val"))
    rng.shuffle(choices)
    out["C-Eval"] = choices

    fact = []
    for r in jsonl_to_list(str(cache / "simplehf/chinese_simpleqa.jsonl")):
        sample = simple.record_to_sample(r)
        prompt = simple.format_prompt_template(sample)
        fact.append(record("Chinese-SimpleQA", r["id"], sample.input,
                           messages=[dict(role="user", content=prompt)], answer=sample.target,
                           source_id=r["id"], category=r["primary_category"], aliases=date_or_title_aliases(sample.target)))
    out["Chinese-SimpleQA"] = frozen_sample(fact, 200, random.Random(SEED))
    instructions = [record("IFEval-ZH", r["key"], r["prompt"], source_id=r["key"],
                           lang=r["lang"], instruction_id_list=r["instruction_id_list"], kwargs=r["kwargs"])
                    for r in jsonl_to_list(str(cache / "if_zh.jsonl"))]
    out["IFEval-ZH"] = frozen_sample(instructions, 200, random.Random(SEED))

    rng = random.Random(SEED)
    tools = []
    for category in ("simple", "multiple"):
        name = f"BFCL_v3_{category}.json"
        raw = jsonl_to_list(str(cache / "bfhf" / name))
        gold = {a["id"]: a["ground_truth"] for a in jsonl_to_list(str(cache / "bfhf/possible_answer" / name))}
        assert {r["id"] for r in raw} == set(gold)
        for r in rng.sample(sorted(raw, key=lambda row: row["id"]), 100):
            turns = r["question"]
            assert len(turns) == len(turns[0]) == 1 and turns[0][0]["role"] == "user"
            messages = [dict(role="system", content=formulate_system_prompt(DEFAULT_SYSTEM_PROMPT_FORMAT, r["function"])),
                        *turns[0]]
            tools.append(record("BFCL-v3", r["id"], turns[0][0]["content"], messages=messages,
                                source_id=r["id"], category=category, functions=r["function"], ground_truth=gold[r["id"]]))
    rng.shuffle(tools)
    out["BFCL-v3"] = tools
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Download missing sources using the pinned manifest")
    args = parser.parse_args()
    for package, expected in [("evalscope","1.15.0"),("bfcl-eval","2025.10.27.1")]:
        if version(package) != expected:
            raise RuntimeError(f"{package}=={expected} required")
    verify_sources(args.cache, args.fetch)
    out = build(args.cache)
    assert {k: len(v) for k,v in out.items()} == EXPECTED
    ids = [r["id"] for rows in out.values() for r in rows]
    assert len(ids) == len(set(ids))
    DEST.mkdir(parents=True, exist_ok=True)
    meta = {}
    for suite, rows in out.items():
        payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",",":"))+"\n" for row in rows).encode()
        (DEST/f"{suite}.jsonl").write_bytes(payload)
        meta[suite] = dict(n=len(rows), sha256=digest(payload), seed=SEED,
                           categories=dict(Counter(r["ref"].get("category", suite) for r in rows)))
        print(suite,len(rows))
    expected = json.loads((PREFLIGHT / "public_frozen.json").read_text())
    if meta != expected:
        raise ValueError("Rebuilt suite differs from the frozen release manifest")


if __name__ == "__main__":
    main()
