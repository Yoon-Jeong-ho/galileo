# Tier-1 experiment plan: cross-family generalization (seeds 1–2)

Goal: reduce reviewer risk by showing GALILEO effects are **not specific to a single model family**.

## Minimal plan (fast, auditable)

- Keep the *exact same protocol* and export schema.
- Run **seeds 1–2** only for the additional family (enough to demonstrate directionality; expand later only if needed).
- Use the same NUM_SAMPLES (or a smaller one if budget-constrained), and log any deviations.

## Candidate families (pick 1 first)

1) **Llama family** (e.g., Llama-3.1-8B-Instruct / Llama-3.3-70B-Instruct)
   - Pros: common baseline; strong reviewer familiarity.
   - Cons: access/weight policy; 70B is heavy.

2) **Mistral family** (e.g., Mistral-7B-Instruct-v0.3, Mixtral 8x7B)
   - Pros: widely used; 7B easy.

3) **EXAONE / other local Korean-friendly family** (if already in infra)
   - Pros: may be easiest if already downloaded.

## What to report (paper-facing)

- Replicate the main views (collapsed is OK for Tier-1):
  - Survival@5 (control vs persona)
  - Fail@1 (control vs persona)
  - Never-fail (control vs persona)
  - (Optional) recovery collapsed
- Produce the same `paper_exports/` bundle and validator OK.

## Concrete run template (to copy into run.log)

- Model: <MODEL_NAME>
- Seeds: 1,2
- GPUs: 4/5/6
- OUT roots:
  - results/<run>_seed1/
  - results/<run>_seed2/
- In each OUT:
  - `paper_exports/` + `metadata.json` + `runner_metadata.json`
  - validator `[OK]` + parity

## Next decision required

Pick **one** model family + exact checkpoint name we can run on `nlp8` with vLLM.
If you tell me what checkpoints already exist on the box (or give access), I’ll write the exact tmux launch commands.
