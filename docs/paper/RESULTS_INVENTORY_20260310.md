# Results Inventory — 2026-03-10 verified package

Repo root: `/data_x/aa007878/projects/galileo`

Purpose: make the currently validated March 10, 2026 results easier to find without changing paper claims or deleting artifacts.

Use this file for:

- locating active evidence roots,
- locating tracked paper-facing exports,
- distinguishing multiseed evidence from directional-only validated runs.

---

## 1) Active validated evidence roots

These are the strongest currently active result roots for the March 10 package.

### Multiseed package (preferred evidence roots)

| Label | Root | Validation signal | Primary tracked exports |
|---|---|---|---|
| evidence multiseed | `tmp/results/qwen7b_evidence_multiseed_gpu5_20260310_231212/` | `GLOBAL_VALIDATE.log` contains `[OK] runner_metadata parity` | `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`, `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv` |
| grounded multiseed | `tmp/results/qwen7b_grounded_multiseed_gpu5_20260310_232747/` | `GLOBAL_VALIDATE.log` contains `[OK] runner_metadata parity` | `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`, `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv` |
| evidence-gate multiseed | `tmp/results/qwen7b_evidencegate_multiseed_gpu5_20260310_234316/` | `GLOBAL_VALIDATE.log` contains `[OK] runner_metadata parity` | `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`, `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`, `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv` |

### Validated directional roots (not preferred as headline evidence)

| Label | Root | Validation signal | Safe use |
|---|---|---|---|
| smoke | `tmp/results/smoke_gpu5_20260310_184715/` | `python3 scripts/validate_paper_exports.py --results_root tmp/results/smoke_gpu5_20260310_184715` → `[OK]` | pipeline sanity / directional context |
| pilot50 math | `tmp/results/pilot50_gpu5_20260310_185825/` | `python3 scripts/validate_paper_exports.py --results_root tmp/results/pilot50_gpu5_20260310_185825` → `[OK]` | directional context only |
| main ARC single-run | `tmp/results/main_arc_gpu6_20260310_191906/` | `python3 scripts/validate_paper_exports.py --results_root tmp/results/main_arc_gpu6_20260310_191906` → `[OK]` | directional context only |

---

## 2) Paper-facing tracked outputs tied to the March 10 package

### Strongest tracked CSVs

- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_evidence_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_grounded_multiseed_seed1-3_deltas_20260310.csv`
- `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_metrics_20260310.csv`
- `docs/paper/artifacts/qwen7b_evidencegate_multiseed_seed1-3_deltas_20260310.csv`

### Additional tracked comparison / figure artifacts

- `docs/paper/artifacts/qwen7b_threeway_multiseed_comparison_20260310.csv`
- `docs/paper/artifacts/qwen7b_tradeoff_gsm8k_multiseed_20260310.svg`
- `docs/paper/artifacts/qwen7b_tradeoff_arc_multiseed_20260310.svg`

Note:

- the metric/delta CSVs have a pinned aggregation script path in `docs/paper/CLAIM_EVIDENCE_MAP.md`;
- the three-way/tradeoff artifacts remain tracked and validated as files, but their exact regeneration provenance is still a documented gap.

---

## 3) Cross-links for maintainers

- claim-level proof mapping:
  - `docs/paper/CLAIM_EVIDENCE_MAP.md`
- strong-vs-weak claim classification:
  - `docs/paper/PAPER_EVIDENCE_STATUS_20260310.md`
- running audit trail:
  - `docs/paper/HEARTBEAT_LOG.md`
- operational SSOT:
  - `README.md`

---

## 4) Current safe housekeeping notes

- No raw March 10 results are currently flagged for deletion here.
- No generated March 10 artifacts are deleted or quarantined by this inventory.
- If future cleanup is needed, prefer quarantining or explicit provenance notes over deletion unless redundancy is obvious and references are audited.
