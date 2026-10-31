"""Build all experiment prompts -> data/prompts.jsonl (seeded, deterministic).

E1 load curve   : k in {1,2,3,5,7,10} random instruction subsets in random order (k=1: every instruction x task).
E2 pairs        : all 45 unordered pairs x 2 orders x 8 tasks.
E3 position     : target first vs last among 5 (same 4 others, same middle order), 10 targets x 40 tasks.
E4 repetition   : k=5 base / target duplicated / neutral filler inserted at the SAME slot, 10 targets x 40 tasks.
"""
import itertools, json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from instructions import IDS, system_prompt

SEED = 42
TASKS = [
    "Explain why the sky is blue.",
    "Give me some tips for learning a new language.",
    "How does a refrigerator keep food cold?",
    "What should I consider when adopting a dog?",
    "Describe how photosynthesis works.",
    "Suggest ways to improve my sleep.",
    "What are the benefits of regular exercise?",
    "How do vaccines help the immune system?",
    "Explain what inflation is.",
    "How can I make my home more energy efficient?",
    "Tell me about the history of the printing press.",
    "What is the difference between weather and climate?",
    "How do I prepare for a job interview?",
    "Explain how a bicycle stays upright while moving.",
    "What are some good habits for managing personal finances?",
    "Why do leaves change color in the fall?",
    "Describe how volcanoes form.",
    "How can I start a small vegetable garden?",
    "What makes a good team leader?",
    "Explain how the internet sends data from one computer to another.",
    "Recommend ways to reduce stress during a busy week.",
    "How do bees make honey?",
    "What is machine learning, in simple terms?",
    "How should I plan a weekend trip to a new city?",
    "Explain why the ocean is salty.",
    "What are some tips for writing a clear email?",
    "How does a rainbow form?",
    "Describe the water cycle.",
    "What should a beginner know about running a first 5k race?",
    "Why is recycling important?",
    "How do airplanes stay in the air?",
    "Give advice for someone moving to a new country.",
    "What causes the seasons on Earth?",
    "How can I become a better public speaker?",
    "Explain what a black hole is.",
    "What are good ways to keep houseplants healthy?",
    "How does compound interest work?",
    "Describe the role of the heart in the human body.",
    "How can students stay focused while studying?",
    "Why do cats purr?",
]
assert len(TASKS) == 40


def main():
    rng = random.Random(SEED)
    rows = []

    def add(exp, cond, task_id, lines, **meta):
        rows.append(dict(pid=f"{exp}-{len(rows):05d}", exp=exp, cond=cond, task_id=task_id, task=TASKS[task_id],
                         lines=lines, system=system_prompt(lines), **meta))

    # E1 load curve
    for i in IDS:
        for t in range(40):
            add("E1", "k1", t, [i], k=1)
    for k in (2, 3, 5, 7, 10):
        for rep in range(10):
            for t in range(40):
                add("E1", f"k{k}", t, rng.sample(IDS, k), k=k)
    # E2 pairs (both orders)
    for pi, (a, b) in enumerate(itertools.combinations(IDS, 2)):
        for oi, order in enumerate(([a, b], [b, a])):
            for j in range(8):
                t = (pi * 16 + oi * 8 + j) % 40
                add("E2", "pair", t, list(order), k=2)
    # E3 position: target first vs last, others identical
    for tgt in IDS:
        for t in range(40):
            others = rng.sample([i for i in IDS if i != tgt], 4)
            add("E3", "first", t, [tgt] + others, k=5, target=tgt)
            add("E3", "last", t, others + [tgt], k=5, target=tgt)
    # E4 repetition with a length-matched filler control
    for tgt in IDS:
        for t in range(40):
            base = rng.sample([i for i in IDS if i != tgt], 4)
            base.insert(rng.randrange(5), tgt)
            slot = rng.randrange(6)
            while slot <= base.index(tgt) + 1 and slot >= base.index(tgt):   # keep the copy non-adjacent
                slot = rng.randrange(6)
            rep, fil = base.copy(), base.copy()
            rep.insert(slot, tgt)
            fil.insert(slot, "FILLER")
            for cond, lines in (("base", base), ("repeat", rep), ("filler", fil)):
                add("E4", cond, t, lines, k=5, target=tgt, slot=slot)

    # Robustness variant: same E1 (k=1,5,10) and E3 designs rendered as a numbered list
    num = [dict(r, pid="N" + r["pid"], variant="numbered", system=system_prompt(r["lines"], numbered=True))
           for r in rows if (r["exp"] == "E1" and r["k"] in (1, 5, 10)) or r["exp"] == "E3"]
    with open(ROOT / "data/prompts_numbered.jsonl", "w") as f:
        for r in num:
            f.write(json.dumps(r) + "\n")
    print("numbered variant:", len(num))

    out = ROOT / "data/prompts.jsonl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(len(rows), Counter((r["exp"], r["cond"]) for r in rows))
    print(rows[3000]["system"], "\nUSER:", rows[3000]["task"])


if __name__ == "__main__":
    main()
