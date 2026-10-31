"""Dump representative failure examples for the strongest interference cells -> results/analysis/error_examples.md"""
import json, random, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instructions import check

CELLS = [("F_title", "F_bullets"), ("P_name", "F_bullets"), ("P_name", "P_ahoy"), ("P_name", "S_lower"),
         ("F_end", "F_bullets"), ("S_nocomma", "F_title")]
MODELS = ["qwen2.5-3b", "llama3.2-3b", "qwen2.5-7b"]


def main():
    P = {json.loads(l)["pid"]: json.loads(l) for l in open(ROOT / "data/prompts.jsonl")}
    G = {}
    for m in MODELS:
        for l in open(ROOT / f"results/gens/{m}.jsonl"):
            r = json.loads(l); G[(m, r["pid"])] = r["text"]
    rng = random.Random(0)
    out = ["# Error examples (E2 pairs; first 500 chars of each response)\n"]
    df = pd.read_csv(ROOT / "results/scores.csv")
    for a, b in CELLS:
        sub = df[(df.exp == "E2") & (df.iid == a) & (df.others == b) & df.model.isin(MODELS)]
        alone = df[(df.exp == "E1") & (df.k == 1) & (df.iid == a) & df.model.isin(MODELS)].ok.mean()
        out.append(f"\n## Target {a} with partner {b}: compliance {100 * sub.ok.mean():.0f}% (alone {100 * alone:.0f}%, "
                   f"models {', '.join(MODELS)})\n")
        want = 0 if a != "S_nocomma" else 1           # S_nocomma is a facilitation cell: show successes
        rows = sub[sub.ok == want]
        for _, r in rows.sample(min(2, len(rows)), random_state=1).iterrows():
            p = P[r.pid]
            out.append(f"**{r.model}**, system lines: {p['lines']}, user: _{p['task']}_\n\n```\n{G[(r.model, r.pid)][:500]}\n```\n")
    (ROOT / "results/analysis/error_examples.md").write_text("\n".join(out))
    print("\n".join(out)[:6000])


if __name__ == "__main__":
    main()
