# Consistency of Large Reasoning Models Under Multi-Turn Attack

- Year: 2026
- Venue: arXiv
- Authors: Yubo Li, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/abs/2602.13093
- BibTeX key (if we add it): li2026consistency
- Tags: multi-turn, adversarial, social-pressure, consistency, flip-metrics, reasoning-models, confidence

## One-sentence takeaway

Frontier “reasoning models” are *more* robust than an instruction-tuned baseline under an 8-round adversarial follow-up protocol, but still flip via identifiable social/psychological failure modes, and confidence-based defenses (CARG) break because reasoning traces induce overconfidence.

## What problem does it solve?

- Characterizes **multi-turn answer stability** of large *reasoning* models under **adversarial conversational pressure**, addressing the question: does chain-of-thought / extended reasoning automatically confer robustness to persuasion / “Are you sure?” style challenges?
- Diagnoses *how* and *why* flips happen (trajectory patterns + failure-mode taxonomy) and tests whether a prior confidence-based mitigation (CARG) transfers to reasoning models.

## What is the core method / protocol?

- **Task/data:** “MT-Consistency” factual multiple-choice set (4 options), across 39 subjects; only sequences where the model’s initial answer is correct are attacked.
- **Protocol:** an **8-round** adversarial follow-up sequence after the initial response, with the **order randomized** via a permutation per run/seed to reduce position effects.
- **Attack types (taxonomy):** includes at least (as described):
  - simple questioning (“Are you sure?”),
  - misleading suggestion proposing a specific wrong alternative,
  - emotional / impolite framing,
  - appeals to expertise/authority,
  - consensus/social pressure.
- **Trajectory analysis:** classifies correctness sequences across turns into patterns (e.g., No Flip, Immediate/Delayed Recovery, Delayed Sustained, Oscillating, Terminal Capitulation, Double Flip).
- **Mitigation test:** applies **Confidence-Aware Response Generation (CARG)** (Li et al., 2025b) by extracting logprob-derived confidence and **embedding confidence** into the dialogue history; compares elicitation strategies (overall vs answer_only vs random).

## What are the key metrics?

- **Initial Accuracy** (Acc_init): correctness before adversarial pressure.
- **Follow-up Accuracy** (Acc_avg): average correctness over adversarial rounds.
- **Position-Weighted Consistency (PWC)** (from Li et al., 2025b): exponentially discounted sum over turn-wise correctness, penalizing early failures more.
- **Trajectory pattern counts** (flip pattern distribution) and **attack-type flip rates** (vulnerability profile).
- **Confidence calibration diagnostics:**
  - point-biserial correlation between confidence and correctness,
  - ROC-AUC of confidence as a correctness classifier,
  - flip rates by confidence terciles.

## What are the main results?

- **Reasoning helps but doesn’t solve robustness:** 8/9 reasoning models show **significantly higher PWC** than the GPT-4o baseline (Welch t-tests), with reported effect sizes d≈0.12–0.40.
- **Universal weak spot:** **misleading suggestions** (explicitly offering an incorrect alternative) are the most effective or second-most effective attack across models.
- **Model-specific social-pressure sensitivity:** consensus/other social cues affect some models disproportionately (notably Claude-4.5 shows high oscillation / instability).
- **Five failure modes** identified via trajectory inspection: **Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, Reasoning Fatigue**; Self-Doubt + Social Conformity account for ~50% of failures.
- **CARG fails on reasoning models:** confidence is poorly predictive of correctness (reported r≈-0.08, ROC-AUC≈0.54; confidence tightly clustered near ~96%), so confidence-aware interventions underperform; surprisingly, **random confidence embedding** can outperform targeted extraction.

## How is this similar to GALILEO?

- Directly targets **multi-turn robustness under social/adversarial pressure** and emphasizes **when flips occur** (PWC / trajectory patterns), aligning with GALILEO’s focus on pressure-driven drift vs stable belief.
- Provides a concrete **attack taxonomy** and **failure-mode labels** that can map onto GALILEO’s planned analysis sections.
- Highlights that “more reasoning” does not automatically prevent persuasion-style failures—supporting the motivation for GALILEO-style evaluation.

## How is this different from GALILEO?

- Uses **factual MCQ** as the primary setting; GALILEO may care about richer belief states, explicit drift-vs-revision controls, and recovery/return-to-truth metrics.
- Focuses heavily on comparing **frontier reasoning models** and diagnosing CARG; less emphasis on designing **controls** that separate evidence-based revision from pure social pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit **neutral re-ask / evidence injection controls**, it can more cleanly identify *pressure-driven drift* versus legitimate revision.
- If GALILEO measures **recovery-after-flip** as a first-class outcome, it can go beyond first-failure / aggregate consistency.

## Where GALILEO is weaker / needs to improve

- GALILEO should match or exceed this paper’s **auditability**:
  - randomized follow-up ordering to reduce position artifacts,
  - reporting both aggregate metrics (Acc_avg/PWC) and **trajectory patterns**.
- GALILEO should anticipate **confidence signal failure** on reasoning models; logprob confidence may be uninformative when long reasoning traces inflate apparent certainty.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a **PWC-style** metric (or an explicit mapping) alongside Turn-of-Failure / survival curves, emphasizing early-vs-late failures and rewarding fast recovery.
- [ ] Include a **misleading-suggestion** pressure operator as a canonical strong adversary baseline; treat it separately from generic disagreement.
- [ ] Report **trajectory patterns** (no flip / recovery / oscillation / terminal capitulation) to distinguish “waver then recover” vs “catastrophic capitulation”.
- [ ] Avoid relying on raw **logprob confidence** for interventions on reasoning models; consider alternative uncertainty signals (self-consistency, verifier-based confidence, calibrated abstention).

## Quotes / details to potentially cite

- Abstract framing: reasoning gives “meaningful but incomplete robustness”; misleading suggestions universally effective; social pressure is model-specific.
- Experimental setup: MT-Consistency MCQ set; **8-round** adversarial follow-ups with randomized order.
- Metrics: Acc_init, Acc_avg, **PWC** with exponential discounting; trajectory patterns; confidence-correctness r≈-0.08 and ROC-AUC≈0.54.
- Identified failure modes: Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, Reasoning Fatigue.
