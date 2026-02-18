# Reflective Confidence: Correcting Reasoning Flaws via Online Self-Correction

- Year: 2025
- Venue: arXiv (cs.AI), under submission
- Authors: Qinglin Zeng; Jing Yang; Keze Wang
- URL: https://arxiv.org/abs/2512.18605
- BibTeX key (if we add it): zeng2025reflectiveconfidence
- Tags: self-correction, confidence, early-stopping, self-consistency, reflection, math-reasoning

## One-sentence takeaway

Use an *online* confidence dip as a trigger to prompt the model to reflect and patch its current chain-of-thought, instead of early-stopping and discarding the partial trajectory.

## What problem does it solve?

- Self-consistency / multi-sample reasoning boosts accuracy but is expensive (token cost scales with number of trajectories).
- Confidence-based early stopping (e.g., DeepConf) saves compute by terminating low-confidence trajectories, but wastes already-spent tokens and may throw away trajectories that are “fixable.”

## What is the core method / protocol?

- Define a token-level confidence score as the mean log-probability over the top-k next-token candidates.
- Smooth it into a “group confidence” via a sliding window average over the last n steps.
- Calibrate an adaptive reflection threshold using a warmup set of initial trajectories: set threshold s to the p-th percentile (paper uses p=10) of the *minimum* group confidence observed in warmup trajectories.
- During generation, when group confidence drops below s (first trigger time i*):
  - Pause decoding.
  - Construct a “reflection prompt” that includes (a) the original question and (b) the partial reasoning up to i*.
  - Ask the model to identify the likely flaw near the end and continue with a corrected trajectory.
  - Splice corrected continuation back and resume generation.

## What are the key metrics?

- Accuracy (%) on AIME 2025.
- Total token cost (reported as “Total Tokens (M)” in their tables).
- In ablation: “Salvage Rate” = fraction of intervened paths that end up correct.

## What are the main results?

- On AIME 2025 with Qwen3-8B, the proposed ReflectiveConf improves the accuracy–compute tradeoff over:
  - Standard self-consistency (cons@K)
  - DeepConf-style online early stopping
- In the high-budget setting (B=32; 16 warmup + 16 reasoning traces), they report ~83.3% accuracy vs ~70.0% for self-consistency, with only a marginal cost increase.
- Ablation: a simple “restart/backtrack and regenerate” intervention (Conf-Restart) is materially worse than guided reflection; reflective prompting yields much higher salvage rate (~65.8% vs ~35.4% in their reported K=32 ablation).

## How is this similar to GALILEO?

- Same high-level theme: use *signals during inference* to steer the reasoning process rather than only ranking/filtering completed outputs.
- Emphasizes online correction and “salvaging” partial work, which aligns with interactive / iterative reasoning systems.

## How is this different from GALILEO?

- This is primarily an inference-time heuristic around confidence thresholds + self-reflection prompting; it is not a new training objective or a discovery/scientific workflow.
- Focuses on math reasoning benchmarks (AIME) and self-consistency sampling, not on broader agentic pipelines or domain-specific evaluation.
- Confidence is defined from token probabilities (top-k mean logprob + sliding-window smoothing), rather than task-specific uncertainty estimation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has principled uncertainty modeling / verification hooks, it may provide stronger guarantees than a percentile-calibrated threshold.
- If GALILEO targets scientific discovery / structured workflows, the scope is broader than “fix chains of thought on math.”

## Where GALILEO is weaker / needs to improve

- If GALILEO currently discards low-confidence trajectories (early stopping, pruning), this paper suggests a cheap alternative: trigger a corrective reflection instead of terminate.
- If GALILEO lacks a robust online “confidence dip” detector, their sliding-window signal + warmup percentile calibration is a simple baseline worth comparing.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a baseline: “confidence dip → reflection prompt → continue” (vs confidence dip → early stop) and compare at matched token budgets.
- [ ] Test warmup-percentile threshold calibration (p=10) vs fixed threshold vs adaptive per-instance calibration.
- [ ] If GALILEO uses multiple samples, report a “salvage rate” analog: among trajectories flagged as low-confidence, what fraction can be recovered by intervention?
- [ ] In related work, position this as: *online self-correction triggered by intrinsic confidence*, contrasted with post-hoc reflection (Self-Refine) and early stopping (DeepConf).

## Quotes / details to potentially cite

- “We introduce Reflective Confidence… transforms a low-confidence signal from a termination symbol into a reflection trigger.”
- Reflection prompt template (paraphrase): confidence drop indicates likely flaw; show partial reasoning; ask to analyze final part, identify error/uncertainty, and continue rigorously with a corrected continuation.
- Setup details: AIME 2025; Qwen3-8B; budgets B=2 and B=32; threshold at 10th percentile of warmup minima.
