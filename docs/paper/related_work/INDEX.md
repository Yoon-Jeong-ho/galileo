# Related Work Index

> Rolling index. Add one line per paper + link to its note.

Columns:
- Area: (sycophancy / truthfulness / multi-turn eval / persona pressure / jailbreak / robustness / calibration / etc.)
- Key hook: what we cite it for (1 phrase)
- Our mapping: which GALILEO concept it supports/contrasts (Survival/TOF/Recovery/Neutral control)

---

## Papers

- sycophancy / multi-turn benchmark — **SYCON-Bench (EMNLP Findings 2025)** — ToF/NoF metrics; map their ToF ↔ our TOF; our delta = ground-truth + recovery + neutral control.  
  - Note: `papers/sycon_bench_2025.md`
- sycophancy / multi-turn benchmark — **TRUTH DECAY (arXiv 2025)** — extended dialogue sycophancy (**numbers extracted**); our delta = neutral control + survival/TOF/recovery on ground-truth tasks.  
  - Note: `papers/truth_decay_2025.md`
- framing / rebuttal — **Challenging the Evaluator (EMNLP Findings 2025)** — follow-up rebuttal framing increases endorsement (**numbers extracted**); useful for rebuttal-framing related work.  
  - Note: `papers/challenging_the_evaluator_2025.md`
- robustness / survival analysis — **Time-To-Inconsistency (arXiv 2025)** — explicit survival/time-to-failure framing in multi-turn robustness; cite as methodological neighbor.  
  - Note: `papers/time_to_inconsistency_2025.md`
- truthfulness / self-verification — **Chain-of-Verification (CoVe) (Findings ACL 2024)** — verify-then-answer style procedure reduces hallucinations; useful contrast for our *dialogue* survival/TOF + recovery framing.  
  - Note: `papers/chain_of_verification_cove_2024.md`
