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
- [ ] For **persona-mechanism claims** (C2-style), ensure the table/figure includes the **Neutral Re-asking Control** as a drift baseline (not persona-only).

### Initial claim→evidence map (draft; fill/iterate)

Claim | Evidence (figure/table) | Tracked artifact(s) | Reproducer / paper-ready run
---|---|---|---
C1 (Dynamics): failures are multi-turn trajectories; survival/TOF needed beyond initial accuracy. | `docs/paper/figures/survival_curves_rounds_seed1-4_20260209.svg`; `docs/paper/figures/tof_personawise_fail1_delta_seed1-4_20260209.svg` | `docs/paper/artifacts/survival_curve_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`; `docs/paper/artifacts/tof_personawise_fail1_never_control_vs_persona_seed1-4_mean_std_20260209.csv` | nlp8: `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` (see `results_paper/GLOBAL_VALIDATE.log`)
C2 (Mechanism vs drift): persona pressure causes effects beyond generic drift; control baseline is essential. | `docs/paper/figures/table_w_effect_delta_seed1-4_20260209.svg` (includes Neutral Re-asking Control as drift baseline) | `docs/paper/artifacts/table_w_effect_delta_seed1-4_20260209.csv`; `docs/paper/artifacts/table_w_control_vs_persona_seed1-4_mean_std_20260209.csv` | generator: `scripts/make_table_w_control_vs_persona.py --control_persona_id neutral_reask_control --round 5`; validate via `results_paper/GLOBAL_VALIDATE.log`
C3 (Robustness vs recovery): recovery@flip is distinct from survival; interventions affect recovery differently. | `docs/paper/figures/recovery_personawise_delta_seed1-4_20260209.svg` (baseline) + cross-ref §7.4 ablation summary | `docs/paper/artifacts/recovery_personawise_control_vs_persona_seed1-4_mean_std_20260209.csv`; `docs/paper/artifacts/tier1_qwen2p5_7b_vta_seed1-2_recovery_collapsed_20260210.csv` | paper-ready runs: `results_paper/qwen_control_seed{1..4}`, `results_paper/qwen_persona_seed{1..4}`, `results_paper/qwen_vta_seed{1,2}`; validate via `scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity`
C4 (Cross-family): effects replicate across model families under the same protocol. | `docs/paper/figures/cross_family_survival_r5_control_vs_logicaltrap_seed1-2_20260212.svg` | `docs/paper/artifacts/tier1_mistral7b_seed1-2_survival_summary_20260210.csv`; `docs/paper/artifacts/tier1_llama3_8b_seed1-2_survival_summary_20260210.csv` | paper-ready runs: `results_paper/mistral_seed{1,2}`, `results_paper/llama_seed{1,2}`; validate via `results_paper/GLOBAL_VALIDATE.log`
C5 (Reproducibility): strict data + paper-ready exports + parity validation. | (protocol+pipeline) `docs/paper/figures/protocol_overview.svg` + validator log | `results/**/paper_exports/{survival_curve.csv,turn_of_failure.csv,flip_samples.csv,metadata.json,runner_metadata.json}` | `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity` (PASS in `results_paper/GLOBAL_VALIDATE.log`)
C6 (Appendix robustness): decoding sensitivity does not qualitatively change persona–control gaps. | Appendix~A.1, Fig.~\ref{fig:decoding-sweep} | `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv` | `results_paper/qwen_temp0_seed{1,2}`, `results_paper/qwen_temp0p7_seed{1,2}`; validate via `results_paper/GLOBAL_VALIDATE.log`

Recommended artifacts to cite in-paper:
- Survival curves: `paper_exports/survival_curve.csv` → plotted figure(s)
  - Include rows for personas **and** the **Neutral Re-asking Control** (drift baseline).
- Turn-of-failure: `paper_exports/turn_of_failure.csv` → table/heatmap
  - Include control vs persona breakdown where relevant (e.g., Fail@1 / Never-fail).
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
- [x] One **decoding sensitivity** sweep (temperature or greedy vs sampling). Completed (Qwen; seed1–2; `--greedy_temperature` 0.0 vs 0.7). Artifact: `docs/paper/artifacts/decoding_sweep_qwen_temp_summary_seed1-2_20260211.csv`; runs: `results_paper/qwen_temp0_seed{1,2}`, `results_paper/qwen_temp0p7_seed{1,2}`; validator: `results_paper/GLOBAL_VALIDATE.log` (all `[OK]`). (Plan/commands: `docs/paper/DECODING_SENSITIVITY_SWEEP.md`)
- [ ] One **recovery prompt ablation** (baseline vs variant).
- [ ] One **qualitative analysis** section with flip examples + taxonomy.
- [ ] Include a **non-persona drift control** (aka **Neutral Re-asking Control**) reported alongside persona curves/tables.

## 4) Reproducibility / engineering checklist
- [x] Single command (or 2–3 commands) to reproduce the main tables/figures.
  - SVG figures from tracked artifacts (stdlib-only): `python3 scripts/make_paper_figures_from_artifacts.py`
  - (Optional for LaTeX) SVG→PDF (no-sudo AppImage path): `bash scripts/get_inkscape_appimage.sh && bash scripts/convert_figures_svg_to_pdf.sh && bash scripts/check_pdf_figures.sh`
  - Paper-ready export validation (paper SSOT): `python3 scripts/validate_paper_exports.py --results_root results_paper --check_runner_parity`
- [ ] Results directories are self-contained and documented (what each CSV/JSONL means).
- [ ] Each reported result bundle has a passing validation log (e.g., `GLOBAL_VALIDATE.log` from `scripts/validate_paper_exports.py --check_runner_parity`).
- [ ] Fixed seeds documented; randomization sources described.
- [ ] Evaluation details: answer extraction rules; normalization; alias handling.
- [ ] Provide a lightweight “smoke test” run (small `NUM_SAMPLES`).
- [ ] Export per-run **metadata** (e.g., `results/<run>/paper_exports/metadata.json`) including decoding params, seed, git commit hash, and condition identifiers (personas + `neutral_reask_control`) so settings are auditable.
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
- [ ] Figure pipeline is unambiguous:
  - [ ] SVG source-of-truth figures exist under `docs/paper/figures/` (generated from tracked artifacts).
  - [ ] Figure→artifact mapping is recorded (see `docs/paper/README.md` and/or `docs/paper/FIGURE_CAPTIONS.md`).
  - [ ] If LaTeX requires PDF, PDFs are generated reproducibly (see `scripts/convert_figures_svg_to_pdf.sh`) and included in the build.
  - [ ] **Build env has SVG→PDF tooling**: either `rsvg-convert` (Ubuntu `librsvg2-bin`) or `inkscape`.
    - No-sudo fallback supported: `scripts/get_inkscape_appimage.sh` + `scripts/convert_figures_svg_to_pdf.sh` (AppImage).
  - [ ] PDF smoke-check passes (header/size): `bash scripts/check_pdf_figures.sh`

## 8) Final pre-submission pass
- [ ] Anonymization (if required): remove identifying paths/names.
  - [ ] Use `docs/paper/ANONYMIZATION_NOTES.md` as the SSOT for what to grep/sanitize/exclude.
  - [ ] Replace absolute paths (e.g., `/mnt/raid6/...`, `/data_x/...`) with generic placeholders.
  - [ ] Remove hostnames / usernames / internal machine identifiers from docs.
  - [ ] Ensure figures/tables do not embed run directory names containing identifying info.
  - [ ] Double-check `results/**/run.log` before packaging any artifact bundle.
- [ ] Camera-ready checklist (after acceptance): acknowledgements, ethics, artifacts.
- [ ] Run spellcheck + consistency checks (persona names, metric names, dataset names).

