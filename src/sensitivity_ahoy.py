"""Sensitivity check for a P_ahoy verifier false negative (post-hoc, found while writing the paper).

instructions._strip_title only drops a title on its own line, so "<<Title>> Ahoy, matey!" scores as a failure.
This re-scores P_ahoy with a same-line title allowed and re-runs the unchanged analyze.py on a copy of the scores.
The original results/ tables are not modified. Output: results/analysis/sensitivity_ahoy/ (+ figures in a subfolder).
Usage: python src/sensitivity_ahoy.py
"""
import json, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
DEST = ROOT / "results/analysis/sensitivity_ahoy"
_LEAD_TITLE = re.compile(r"^[\s*#_]*<<[^<>\n]+>>[\s*#_:\-–—]*")


def ahoy_fixed(text):
    m = re.search(r"[a-z]+", _LEAD_TITLE.sub("", text.strip()).lower())
    return bool(m) and m.group(0) == "ahoy"


def main():
    df = pd.read_csv(ROOT / "results/scores.csv")
    new = {}
    for f in (ROOT / "results/gens").glob("*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            new[(f.stem, r["pid"])] = int(ahoy_fixed(r["text"]))
    mask = df.iid == "P_ahoy"
    fixed = [new[k] for k in zip(df.model[mask], df.pid[mask])]
    changed = df.loc[mask].assign(new=fixed)
    print(changed.assign(flip=changed.new - changed.ok).groupby(["model", "flip"]).size().unstack(fill_value=0))
    df.loc[mask, "ok"] = fixed
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "results").mkdir()
        shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data", tmp / "data")
        df.to_csv(tmp / "results/scores.csv", index=False)
        subprocess.run([sys.executable, str(tmp / "src/analyze.py")], check=True, cwd=tmp, stdout=open(tmp / "log.txt", "w"))
        shutil.rmtree(DEST, ignore_errors=True)
        shutil.copytree(tmp / "results/analysis", DEST)
        shutil.copytree(tmp / "figures", DEST / "figures")


if __name__ == "__main__":
    main()
