"""Score every generation against each (distinct) instruction in its system prompt.

Output: results/scores.csv, one row per (model, prompt, instruction) with compliance `ok` (0/1) plus design metadata.
"""
import json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instructions import check, GROUP, INSTRUCTIONS
from models import MODELS


def main():
    prompts = {}
    for fn in ("prompts.jsonl", "prompts_numbered.jsonl"):
        prompts.update({p["pid"]: p for p in map(json.loads, open(ROOT / "data" / fn))})
    rows, summary = [], []
    for key in MODELS:
        f = ROOT / "results/gens" / f"{key}.jsonl"
        if not f.exists():
            continue
        gens = {}
        for l in open(f):
            r = json.loads(l)
            gens[r["pid"]] = r                     # last write wins (resumed runs never duplicate, but be safe)
        summary.append((key, len(gens), sum(g["finish"] == "length" for g in gens.values())))
        for pid, g in gens.items():
            p = prompts[pid]
            lines = [x for x in p["lines"] if x != "FILLER"]
            for iid in dict.fromkeys(lines):         # distinct, in order of first appearance
                pos = p["lines"].index(iid)          # 0-based slot of first occurrence (fillers count as slots)
                rows.append(dict(model=key, pid=pid, exp=p["exp"], cond=p["cond"], task_id=p["task_id"], k=p["k"],
                                 target=p.get("target"), iid=iid, cat=INSTRUCTIONS[iid]["cat"], group=GROUP[iid],
                                 pos=pos, n_lines=len(p["lines"]), n_occ=p["lines"].count(iid),
                                 is_target=(iid == p.get("target")), ok=int(check(iid, g["text"])),
                                 truncated=int(g["finish"] == "length"), n_out=g["n_out"],
                                 others="|".join(x for x in dict.fromkeys(lines) if x != iid)))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/scores.csv", index=False)
    for s in summary:
        print("%-14s gens=%5d truncated=%4d" % s)
    print(df.groupby(["model", "exp"]).ok.mean().unstack().round(3))


if __name__ == "__main__":
    main()
