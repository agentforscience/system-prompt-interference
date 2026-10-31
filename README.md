# System Prompt Interference Patterns

A controlled study of how multiple verifiable instructions in an LLM **system prompt** compete with each other. It
tests four factors: instruction count, pairwise category interference, position (first vs last), and repetition (with
a length-matched filler control). It uses 8 open-weight chat models from 0.5B to 8B (Qwen2.5, Llama-3.x, Qwen3), with
5,120 prompts per model and deterministic IFEval-style verifiers. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **Count dominates.** Mean per-instruction compliance is 83% with 1 instruction, 70% with 5 and 59% with 10. The
  rate of satisfying *all* instructions collapses: 83% → 37% (5) → 5% (10). The best model (Qwen3-8B) satisfies all 10
  only 24% of the time.
- **"Single instruction > 90%" holds for only 4 of 8 models.** Qwen2.5-0.5B and 1.5B are at 43% and 60%.
- **Interference is specific, not categorical.** Format/length partners hurt persona/style instructions 5–12 pp more
  than persona/style partners do, in 4 of 8 models. The 20–30 pp drops appear only in tiny models. The biggest effect
  is *layout capture*: "exactly 3 bullets" erases the closing phrase (−30 pp), keywords (−20) and self-introductions.
  Some pairs help each other (no-comma rule +5–11 pp next to structural instructions).
- **Position and repetition are weak levers.** Placing an instruction last instead of first gives +1.7 pp overall
  (Qwen +5 to +8; Llama ≈ 0). Repeating it gives +3.3 pp. The −0.9 pp cost to the other instructions also appears with a
  neutral filler line, so it comes from prompt length, not repetition.
- **Size and decay depend on the scale used.** Smaller models lose a larger *share* of their compliance (ρ = −0.95).
  In absolute pp there is no size trend, and in odds terms the 8B models decay at least as steeply.

## Reproduce
```bash
uv venv --python 3.12 && source .venv/bin/activate && uv sync        # or: uv pip install -r requirements.txt
export NUMPY_MADVISE_HUGEPAGE=0 HF_HOME=/data/transformers HF_HUB_OFFLINE=1
python src/build_prompts.py                         # -> data/prompts.jsonl, data/prompts_numbered.jsonl
# local models (vLLM, one GPU; the queue file lists model keys from src/models.py)
printf "qwen2.5-0.5b\nqwen2.5-1.5b\nqwen2.5-3b\nllama3.2-1b\nllama3.2-3b\nqwen2.5-7b\nllama3.1-8b\n" > logs/queue.txt
CUDA_VISIBLE_DEVICES=0 python src/worker.py logs/queue.txt 9.0       # stop with: touch logs/STOP_workers
python src/gen_server.py 48                         # Qwen3-8B via an OpenAI-compatible vLLM server on :8011
VARIANT=num python src/gen_server.py 48             # numbered-list robustness variant
python src/score.py && python src/analyze.py && python src/examples.py
```
Generation takes ~3–15 min per model on an A6000 slice. Scoring and analysis take ~2 min, and the analysis is
deterministic (seeded bootstrap).

## Files
| path | contents |
|---|---|
| `planning.md` | motivation, hypothesis decomposition, pre-registered analysis plan |
| `src/instructions.py` | 10-instruction pool, verifiers, system-prompt renderer (self-test: `python src/instructions.py`) |
| `src/build_prompts.py` | E1–E4 design and robustness variant (seed 42) |
| `src/worker.py`, `src/models.py` | in-process vLLM generation worker for shared GPUs; model registry |
| `src/gen_server.py` | stdlib client for the Qwen3-8B server |
| `src/score.py`, `src/analyze.py`, `src/examples.py` | scoring, statistics and figures, error examples |
| `results/gens/` | raw generations (JSONL per model) |
| `results/scores.csv` | one row per (model, prompt, instruction) |
| `results/analysis/` | all result tables, `summary.json`, `error_examples.md` |
| `figures/` | load curves, interference matrices, position/repetition, size-vs-slope |
