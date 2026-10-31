# Planning: System Prompt Interference Patterns

## Motivation & Novelty Assessment

### Why This Research Matters
Production system prompts routinely stack many requirements (persona, tone, formatting, length limits, sign-offs,
banned punctuation). Developers lack quantitative guidance on *which* instructions crowd each other out, whether
order matters, and whether "say it twice" is a free lunch. Measured interference patterns translate directly into
prompt-writing rules (what to put last, what not to combine, when to split prompts).

### Gap in Existing Work
(No resource-finder output was available for this workspace — no papers/, datasets/, code/ — so this is based on
known literature.) IFEval (Zhou et al., 2023) introduced verifiable instructions, but places 1–3 instructions in the
*user* turn and reports aggregate accuracy. Multi-constraint benchmarks (FollowBench, Jiang et al. 2024; ComplexBench,
Wen et al. 2024; "How many instructions can LLMs follow at once?" / ManyIFEval, Harada et al. 2025) show that the
all-instructions-satisfied rate decays with instruction count. Position effects are documented for long contexts
("Lost in the Middle", Liu et al. 2024). What is missing is a *controlled, factorial* study of instructions in the
**system prompt** that separates (a) count effects, (b) pairwise category interference, (c) position (first vs last),
and (d) repetition (target boost vs collateral cost, with a length-matched control), across model sizes.

### Our Novel Contribution
A fully verifiable 10-instruction pool (4 format/length, 3 persona, 3 style), all mutually satisfiable, used in four
controlled experiments over 6–7 open models (0.5B–8B) to estimate: compliance-vs-load curves, a 10×10 pairwise
interference matrix, paired position effects, and repetition effects with a filler control.

### Experiment Justification
- **E1 Load curve** (k = 1,2,3,5,7,10 random instructions): tests "≥90% single → 50–65% at 5+" and "smaller models
  decay more steeply". Needed because count is the primary independent variable.
- **E2 Pairwise interference** (all 45 pairs × both orders): tests "format/length interfere with persona/style
  (20–30% drop)". Only exhaustive pairs give a clean A-given-B vs A-alone contrast per category.
- **E3 Position** (target first vs last among 5, same other 4 instructions, paired): tests "last > first by 15–25 pp".
- **E4 Repetition** (k=5; baseline vs target duplicated vs neutral filler of equal slot count): tests "+10–15 pp for
  the repeated instruction, −5–8 pp for others", while separating repetition from mere prompt lengthening.

## Research Question
When several verifiable instructions are placed together in an LLM system prompt, how does per-instruction compliance
depend on the number of instructions, their category pairing, their position, and repetition — and does this depend
on model size?

## Hypothesis Decomposition
- H1: single-instruction compliance > 90% (per model, averaged over instructions); with k ≥ 5 per-instruction
  compliance falls to 50–65%.
- H2: compliance of persona/style instructions drops 20–30 pp when paired with a format/length instruction (vs alone),
  more than when paired with another persona/style instruction.
- H3: target compliance when placed last exceeds when placed first by 15–25 pp.
- H4: duplicating an instruction raises its compliance by 10–15 pp and lowers the others' by 5–8 pp.
- H5: decay slope (compliance vs k) is steeper for smaller models.

## Proposed Methodology

### Approach
No API keys are available on this host (OPENROUTER_KEY unset), so we use *real* open-weight models run locally with
vLLM (greedy decoding) plus a host-provided Qwen3-8B vLLM server. Models: Qwen2.5-Instruct 0.5B/1.5B/3B (/7B if GPU
memory allows), Llama-3.2-Instruct 1B/3B, Qwen3-8B (non-thinking). Same family at several sizes gives a clean size axis.

Instruction pool (checked by deterministic Python verifiers, IFEval-style):
| id | category | instruction (abridged) | check |
|---|---|---|---|
| F_title | format | title in <<double angular brackets>> | regex `<<[^>]+>>` |
| F_bullets | format | exactly 3 markdown bullets "* " | count of lines starting `* `/`- ` == 3 |
| F_end | format | end with exact phrase "Any other questions?" | stripped text endswith (case-insens.) |
| L_max | length | fewer than 100 words | word count < 100 |
| P_ahoy | persona | pirate; begin response with "Ahoy" | first word (after optional title line) == ahoy |
| P_name | persona | introduce yourself as Captain Byte | "captain byte" in text |
| P_matey | persona | address the user as "matey" | "matey" in text |
| S_lower | style | all lowercase letters | no uppercase chars |
| S_nocomma | style | no commas | "," not in text |
| S_keyword | style | use word "remarkable" ≥ 2 times | count ≥ 2 |
All 10 are jointly satisfiable (case-insensitive checks for phrases so S_lower doesn't conflict).

User turns: 40 hand-written, diverse, neutral requests (explanations, advice, short how-tos) that don't themselves
impose format constraints.

### Experimental Steps
1. Build prompts (seeded), save to data/prompts.jsonl with full metadata (instruction ids, order, condition).
2. Generate with each model (greedy, max 512 tokens, chat template with system role; Gemma-style models without system
   role excluded).
3. Score every (response, instruction) pair with the verifiers; record truncation.
4. Analyze per hypothesis with item-level mixed/cluster-robust statistics.

### Baselines
- k = 1 (instruction alone) is the baseline for E1/E2.
- E3: paired first-vs-last with identical instruction sets.
- E4: baseline k=5 and a length-matched neutral-filler control ("Be accurate and helpful.").

### Evaluation Metrics
Per-instruction compliance (binary, verifier), all-satisfied rate (prompt-level), and deltas in percentage points.

### Statistical Analysis Plan
- Bootstrap 95% CIs over user tasks (cluster = task) for rates and differences.
- Paired tests: McNemar / sign-flip permutation for E3/E4 paired items.
- E1 slope: logistic GLM compliance ~ k × model (cluster-robust by task); size effect via slope vs log params.
- E2: category-level contrasts with permutation tests; Holm correction across the 5 hypotheses' primary tests.
- α = 0.05.

## Expected Outcomes
Support: numbers in hypothesis ranges. Refutation: e.g., single-instruction compliance far < 90% for small models,
no first/last difference, or repetition gains without collateral costs.

## Timeline and Milestones
Setup/install (~20 min, in background) → prompts + server generation (~30 min) → local models sequentially
(~1–2 h depending on GPU contention) → scoring/analysis (~45 min) → report (~30 min).

## Potential Challenges
- Shared GPUs (memory races): in-process vLLM worker that holds memory across models; run small models first.
- Verifier edge cases (e.g., markdown bold title, bullets "-" vs "*"): audit samples, keep checks lenient-but-fixed.
- Ceiling/floor effects for some instructions: report per-instruction results too.
- Truncation at 512 tokens can break F_end: record finish_reason, report sensitivity excluding truncated outputs.

## Success Criteria
All four experiments run on ≥5 models with real generations; every hypothesis component receives a quantitative
verdict with CIs.
