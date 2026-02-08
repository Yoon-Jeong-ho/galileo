# EMNLP Main Submission Checklist (GALILEO)

> Purpose: keep the paper + repo aligned with **EMNLP Main** expectations. This is a living checklist; check items off as they become *done and verifiable*.

## 0) One-line paper pitch
- [ ] **Problem:** multi-turn belief-consistency failures under social/rhetorical pressure on *ground-truth* tasks.
- [ ] **Solution:** GALILEO protocol + benchmark + reproducible pipeline.
- [ ] **Key novelty:** survival curves + turn-of-failure + recovery, unified across tasks/personas.

## 1) Claims → evidence map (make reviewer verification easy)
For each core claim in the Abstract/Intro:
- [ ] Name the claim (1 sentence).
- [ ] Point to **one table/figure** that supports it.
- [ ] Point to **one script/output path** that reproduces it.

Recommended artifacts to cite in-paper:
- Survival curves: `paper_exports/survival_curve.csv` → plotted figure(s)
- Turn-of-failure: `paper_exports/turn_of_failure.csv` → table/heatmap
- Recovery: `recovery_accuracy.csv` (per run) and aggregated multi-seed tables

## 2) Paper structure sanity (EMNLP Main)
- [ ] Abstract: 1) problem 2) what we do 3) key results 4) why it matters.
- [ ] Intro: clear evaluation gap; crisp contributions list.
- [ ] Method: protocol diagram (3 phases) + personas + recovery prompts.
- [ ] Metrics: define **InitialAcc / Survival(p,r) / TOF / Recovery**.
- [ ] Experiments: datasets, models, decoding, seeds, compute.
- [ ] Results: persona ranking, dataset/task effects, robustness vs recovery.
- [ ] Analysis: failure-mode taxonomy with examples + discussion.
- [ ] Limitations + Ethics: concrete, not boilerplate.

## 3) Experimental minimal set (submission-credible)
- [ ] **At least 2 model families** (ideally 3) evaluated with the same protocol.
- [ ] **Multi-seed** results for the main model(s) (report mean±std).
- [ ] One **decoding sensitivity** sweep (temperature or greedy vs sampling).
- [ ] One **recovery prompt ablation** (baseline vs variant).
- [ ] One **qualitative analysis** section with flip examples + taxonomy.
- [ ] Include a **non-persona drift control** (aka **Neutral Re-asking Control**) reported alongside persona curves/tables.

## 4) Reproducibility / engineering checklist
- [ ] Single command (or 2–3 commands) to reproduce the main tables/figures.
- [ ] Results directories are self-contained and documented (what each CSV/JSONL means).
- [ ] Fixed seeds documented; randomization sources described.
- [ ] Evaluation details: answer extraction rules; normalization; alias handling.
- [ ] Provide a lightweight “smoke test” run (small `NUM_SAMPLES`).
- [ ] Automation hygiene (if using OpenClaw heartbeat updates):
  - [ ] Ensure `HEARTBEAT.md` is **not empty** (not just headers/blank lines), otherwise heartbeats may be skipped.
  - [ ] Avoid OK-only replies (e.g., `HEARTBEAT_OK`) if you expect a DM every tick.

## 5) Dataset + licensing + release
- [ ] For each dataset: **license/terms** noted; download instructions.
  - [ ] GSM8K: license/terms + citation noted.
    - Source: https://github.com/openai/grade-school-math
    - License: MIT (see https://github.com/openai/grade-school-math/blob/master/LICENSE)
  - [ ] SVAMP: license/terms + citation noted.
    - Source: https://github.com/arkilpatel/SVAMP
    - License: MIT (see https://github.com/arkilpatel/SVAMP/blob/main/LICENSE)
  - [ ] ARC / SQuAD / TriviaQA (if used in main results): license/terms + citation noted.
    - ARC (AI2 Reasoning Challenge):
      - Dataset card: https://huggingface.co/datasets/allenai/ai2_arc
      - License: CC BY-SA 4.0 (as listed on the dataset card)
    - SQuAD:
      - Source: https://rajpurkar.github.io/SQuAD-explorer/ (check dataset terms/license on the official site)
    - TriviaQA:
      - Source: https://nlp.cs.washington.edu/triviaqa/ (check dataset terms/license on the official site)
- [ ] Any derived/preprocessed artifacts: documented generation steps (script + exact command).
- [ ] If releasing new data (personas/prompts/annotations/taxonomy labels): include license + intended use.
- [ ] Usage notes: clarify what is redistributed vs what users must download themselves.

## 6) Ethics & safety
- [ ] State what the personas simulate (pressure tactics) and what they do *not*.
- [ ] Potential misuse: adversarial prompting; persuasion; model manipulation.
- [ ] Mitigations: release considerations; responsible framing.

## 7) Presentation quality
- [ ] One protocol figure (clean, readable at 1-column width).
- [ ] One main result figure: survival curves (persona-wise).
- [ ] One table: turn-of-failure distribution or Fail@1 / Never-fail.
- [ ] One table: recovery conditional on flipping.
- [ ] Captions explain metrics without forcing readers into the appendix.

## 8) Final pre-submission pass
- [ ] Anonymization (if required): remove identifying paths/names.
  - [ ] Replace absolute paths (e.g., `/mnt/raid6/...`, `/data_x/...`) with generic placeholders.
  - [ ] Remove hostnames / usernames / Slack IDs / internal machine identifiers from logs and docs.
  - [ ] Ensure figures/tables do not embed run directory names containing identifying info.
  - [ ] Double-check `results/**/run.log` before packaging any artifact bundle.
- [ ] Camera-ready checklist (after acceptance): acknowledgements, ethics, artifacts.
- [ ] Run spellcheck + consistency checks (persona names, metric names, dataset names).

