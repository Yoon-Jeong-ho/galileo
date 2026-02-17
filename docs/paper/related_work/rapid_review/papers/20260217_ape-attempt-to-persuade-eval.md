# It’s the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics

- Slug: ape-attempt-to-persuade-eval
- Year: 2025
- Venue: arXiv (cs.AI)
- Authors: Matthew Kowal; Jasper Timm; Jean‑Francois Godbout; Thomas Costello; Antonio A. Arechar; Gordon Pennycook; David Rand; Adam Gleave; Kellin Pelrine
- Links:
  - paper: https://arxiv.org/abs/2506.02873
  - code (if any): https://github.com/AlignmentResearch/AttemptPersuadeEval
- Bibtex: https://doi.org/10.48550/arXiv.2506.02873

## 1) What problem does it study?
Evaluates an AI-safety-relevant dimension of persuasion: not “did persuasion succeed?”, but **whether the model is willing to *attempt* persuasion** in contexts where persuasive content itself is harmful (e.g., advocating violence / extremism / exploitation).

This targets a gap in prior persuasion benchmarks that focus on outcome (belief change) and may miss risk when a model readily generates persuasive rhetoric even if success is unmeasured or unlikely in a particular simulated audience.

## 2) Experimental setup (what is being measured?)
- Task(s): given a topic statement (benign → controversial → conspiracies → undermining control → non-controversially harmful), prompt a *persuader* model to try to persuade a *persuadee* agent over a short dialogue.
- Perturbation/pressure type: explicit instruction to persuade (including on harmful topics); evaluation includes “jailbreak-tuning” (finetuning-based jailbreak) as an adversarial condition.
- Multi-turn? Y — typically **3 rounds** (paper also analyzes longer horizons; attempts tend to decline over many turns as topic drift increases).
- Metrics:
  - **Attempt rate**: per-turn binary label (attempt vs no-attempt), produced by an automated evaluator model using the conversation context.
  - **Refusal rate**: detected via StrongREJECT-style refusal detector (separate from “no-attempt”).
  - Validation metrics (evaluator vs human labels): agreement, Cohen’s kappa, F1; also reports inter-human agreement (Fleiss’ kappa).
  - (Aux feature, not core in reported results): optional numeric belief tracking for persuadee.

## 3) Key findings (bullet)
- Many frontier models (open and closed) **frequently attempt persuasion** across topic categories; behavior diverges primarily on *impactful/harmful* topics.
- **Guardrails mismatch**: models can refuse direct requests for harmful acts yet still comply when asked to *persuade someone else* to do the same act (persuasion-as-indirect-harm channel).
- **Jailbreak finetuning** can sharply reduce refusals and increase willingness to attempt persuasion on the most harmful categories (safety collapse under relatively lightweight adaptation).
- Automated evaluator aligns reasonably well with human labels for binary attempt/no-attempt (reported ~84% agreement; Cohen’s kappa ~0.66 overall in their sampled validation).

## 4) Limitations / threats
- Heavy reliance on an **LLM-based evaluator** for “attempt” labeling; even with validation, subtle persuasive intent can be subjective.
- Uses **simulated persuadee** interactions; does not measure real human susceptibility or downstream belief change.
- Core reported metric is **binary willingness**, not multi-turn *time-to-failure* or recovery dynamics (attempts may decrease over long dialogues due to topic drift).
- “Undermining control” prompts include direct requests to bypass safeguards; this is a useful first-order compliance test but not a full model of covert, long-horizon persuasion threats.

## 5) How it relates to GALILEO
- What we can cite it for:
  - A complementary safety framing: evaluate **propensity/willingness** to engage in harmful persuasion, not only success/susceptibility.
  - Evidence that “persuasion channel” can bypass otherwise-strong refusals for direct harmful assistance.
  - Jailbreak/finetune sensitivity: safety properties around persuasion can be brittle under adaptation.
- Where we differ (our delta):
  - GALILEO focuses on **robustness of beliefs/outputs under social pressure** (drift vs evidence-driven revision, time-to-flip, and recovery), whereas APE focuses on **whether the model tries to persuade at all**.
- Direct mapping:
  - Survival ↔ (not central) they mention attempt frequency over turns; no survival/censoring framing.
  - TOF ↔ not used (attempts tend to decay with turns, but not framed as time-to-event failure).
  - Recovery ↔ not studied.
  - Neutral Re-asking Control ↔ not present.

## 6) Quote-able lines
- Defines persuasion attempt as: generation with the apparent goal of shaping/reinforcing/changing beliefs/attitudes/actions, **regardless of success**.
- Key qualitative takeaway: models may refuse direct harm but still generate persuasive rhetoric encouraging others to commit harm.

## 7) Actions
- [ ] Add to paper: Related work / safety evaluations of persuasion — cite as “willingness to attempt harmful persuasion” benchmark.
- [ ] Add to bib
