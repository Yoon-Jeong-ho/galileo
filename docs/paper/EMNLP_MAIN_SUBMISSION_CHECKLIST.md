# EMNLP Main Submission Checklist (GALILEO)

> Purpose: keep the paper + repo aligned with **EMNLP Main** expectations. This is a living checklist; check items off as they become *done and verifiable*.

## 0) One-line paper pitch
- [ ] **Problem:** multi-turn belief-consistency failures under social/rhetorical pressure on *ground-truth* tasks.
- [ ] **Solution:** GALILEO protocol + benchmark + reproducible pipeline.
- [ ] **Key novelty:** survival curves + turn-of-failure + recovery, unified across tasks/personas.

## 1) Claims → evidence map (make reviewer verification easy)
**SSOT:** `docs/paper/CLAIM_EVIDENCE_MAP.md`

For each core claim in the Abstract/Intro:
- [ ] Name the claim (1 sentence).
- [ ] Point to **one paper-facing figure/table** (link to the exact `docs/paper/figures/*.svg`).
- [ ] Point to **one tracked artifact** (`docs/paper/artifacts/*.csv`) that the figure/table was generated from.
- [ ] Point to **one script + command** that regenerates the artifact/figure.
- [ ] For **persona-mechanism claims** (C2-style), ensure the evidence includes the **Neutral Re-asking Control** as a drift baseline (not persona-only).

(Do not duplicate the full mapping table here; keep it in `CLAIM_EVIDENCE_MAP.md` to avoid drift.)

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
- [ ] Citation hygiene: all `\\cite{...}` keys in the paper drafts exist in `references.bib`.
  - Quick check: `bash scripts/check_citations_vs_bib.sh`
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
    - SQuAD (v1.1):
      - Homepage: https://rajpurkar.github.io/SQuAD-explorer/
      - License: **CC BY-SA 4.0**
        - Hugging Face dataset card front-matter shows `license: cc-by-sa-4.0`: https://huggingface.co/datasets/rajpurkar/squad/raw/main/README.md
      - Citation: Rajpurkar et al., 2016 (EMNLP) — https://arxiv.org/abs/1606.05250
    - TriviaQA:
      - Homepage: https://nlp.cs.washington.edu/triviaqa/
      - Licensing note: the homepage states **“The University of Washington does not own the copyright of the questions and documents included in TriviaQA.”** Treat as **license/terms = unclear/unknown**, and avoid redistributing underlying documents unless the source terms are satisfied.
      - Hugging Face dataset card front-matter currently lists `license: unknown`: https://huggingface.co/datasets/mandarjoshi/trivia_qa/raw/main/README.md
      - Citation: Joshi et al., 2017 — https://arxiv.org/abs/1705.03551
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
- [ ] **Aggregation clarity (avoid Table~W confusion):**
  - [ ] Table~W caption states whether it is **matched persona-wise** vs **pooled across personas**.
  - [ ] If pooled, caption specifies the weighting (e.g., weights proportional to \(|C_p|\) or uniform over personas/examples).
  - [ ] For persona-wise rows, report \(|C_p|\) (or initial-correct count) so reviewers can interpret control baselines that differ by persona.
  - [ ] **Paste-ready Table~W caption template (recommended; matches the current Table-W artifact schema):**
    > *Table W: Control vs persona pressure on initially-correct examples.* For each persona \(p\), we compute metrics on that persona’s initially-correct subset \(C_p\) and evaluate both arms (persona pressure and Neutral Re-asking Control drift baseline) on the same \(C_p\) (matched). We then aggregate persona pressure across personas in two ways: **(i) weighted** pooling with weights \(w_p\propto |C_p|\) (headline) and **(ii) unweighted** mean across personas (each persona counts equally; reported for transparency). Values are mean±std across seeds. Control values reflect the matched per-persona control baselines, not a single global control run.
  - [ ] **Paste-ready persona-wise delta caption note (for persona-wise tables/figures):**
    > Control values are computed on the **same initially-correct subset** as the corresponding persona arm (matched \(C_p\)). Because \(C_p\) can differ by persona, control baselines can differ across personas; within-persona gaps are apples-to-apples.
  - [ ] If you **omit** the unweighted aggregate in the paper table for space, state explicitly in the caption that the table reports the **weighted** (pooled) aggregate only, and note that an unweighted version is available in the tracked Table-W artifact CSV.
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
