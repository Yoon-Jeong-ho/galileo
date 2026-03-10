# Paper Evidence Status — 2026-03-10 verified package

Purpose: classify the currently verified evidence into:

1. **reproducible supported claims**
2. **promising but insufficient observations**
3. **missing-evidence gaps**

This file is for paper-facing evidence triage. It is intentionally more conservative than README prose.

---

## 1) Reproducible supported claims

These are the safest current claims to reuse in paper-facing drafting because they are backed by validated multiseed exports.

### S1. Under the current Qwen7B protocol, authority pressure reduces Survival@5 relative to control on both GSM8K and ARC-Easy.

Supported by validated multiseed roots:

- evidence baseline root:
  - `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/`
  - validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/GLOBAL_VALIDATE.log`
- grounded baseline root:
  - `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/`
  - validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/GLOBAL_VALIDATE.log`

Tracked exports:

- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`

Concrete verified numbers:

- evidence / GSM8K: authority Survival@5 `0.681467`, control `0.896600`, Δ `-0.215133`
- evidence / ARC-Easy: authority Survival@5 `0.285700`, control `0.959200`, Δ `-0.673500`
- grounded / GSM8K: authority Survival@5 `0.703800`, control `0.912267`, Δ `-0.208467`
- grounded / ARC-Easy: authority Survival@5 `0.333333`, control `0.945600`, Δ `-0.612267`

### S2. In the current Qwen7B multiseed package, the pressure effect is larger on ARC-Easy than on GSM8K.

Supported by the same validated multiseed delta exports:

- evidence deltas:
  - ARC-Easy `-0.673500` vs GSM8K `-0.215133`
  - source: `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- grounded deltas:
  - ARC-Easy `-0.612267` vs GSM8K `-0.208467`
  - source: `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`

### Safe body-ready wording candidates

These are drafting candidates, not yet inserted into the paper body:

- “In the current Qwen2.5-7B-Instruct multiseed package, authority pressure reduces Survival@5 relative to the matched re-asking control on both GSM8K and ARC-Easy under both evidence-bearing and grounded-correction baselines.”
- “The same-model effect is materially larger on ARC-Easy than on GSM8K in the March 10, 2026 multiseed package.”

Guardrail:

- keep these as **Qwen7B package claims**, not broader cross-family claims.

---

## 2) Promising but insufficient observations

These observations are validated, but they should not be promoted to headline claims yet.

### P1. Evidence-gate looks promising, but should remain a trade-off observation.

Validated multiseed root:

- `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/`
- validator: `/data_x/aa007878/projects/galileo/tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/GLOBAL_VALIDATE.log`

Tracked exports:

- `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
- `docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
- `docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`

Verified numbers:

- evidence-gate / GSM8K: authority Survival@5 `0.714300`, control `0.920667`, Δ `-0.206367`
- evidence-gate / ARC-Easy: authority Survival@5 `0.360567`, control `0.952400`, Δ `-0.591833`

Why still insufficient for a headline claim:

- it is currently a **single-model** mitigation package;
- the three-way CSV/SVG outputs are tracked artifacts, but their exact regeneration path is not yet pinned in the repo;
- the README itself frames this as a trade-off, not a solved mitigation.

### P2. Smoke / pilot / single-run validated roots are directional only.

Validated roots:

- smoke:
  - `/data_x/aa007878/projects/galileo/tmp/results/smoke_gpu5_20260310_184715/`
  - validator rerun: `python3 scripts/validate_paper_exports.py --results_root tmp/results/smoke_gpu5_20260310_184715`
- pilot50:
  - `/data_x/aa007878/projects/galileo/tmp/results/pilot50_gpu5_20260310_185825/`
  - validator rerun: `python3 scripts/validate_paper_exports.py --results_root tmp/results/pilot50_gpu5_20260310_185825`
- main ARC single-run:
  - `/data_x/aa007878/projects/galileo/tmp/results/main_arc_gpu6_20260310_191906/`
  - validator rerun: `python3 scripts/validate_paper_exports.py --results_root tmp/results/main_arc_gpu6_20260310_191906`

Use:

- safe as pipeline-complete or directional context;
- unsafe as primary headline evidence compared with the multiseed package above.

---

## 3) Missing-evidence gaps

These are the main gaps that would justify a next experiment rather than another documentation-only cycle.

### G1. Regeneration provenance gap for the three-way comparison / tradeoff artifacts

Needed before those artifacts support headline text:

- a pinned script path (or explicit manual provenance note) for:
  - `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
  - `docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
  - `docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`

### G2. Broader evidence is still needed before turning the current Qwen7B mitigation package into a general claim.

Current safe scope:

- repo-level Qwen7B package claim

Not yet justified:

- broad model-family mitigation claim
- broad “evidence-gate solves pressure-induced truth erosion” claim

### G3. If the paper body is to be updated later, A4/A5 need exact draft-section anchors first.

Current safe state:

- the wording candidates above are draft-safe notes only;
- `docs/paper/CLAIM_EVIDENCE_MAP.md` correctly keeps them at repo-level evidence status until a stable draft section exists.
