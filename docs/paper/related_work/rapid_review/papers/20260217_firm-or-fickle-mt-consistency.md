# Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions

- Year: 2025
- Venue: ACL 2025 (arXiv)
- Authors: Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman
- URL: https://arxiv.org/abs/2503.22353
- BibTeX key (if we add it): li2025firm
- Tags: multi-turn, consistency, misleading-followups, social-pressure, recovery-metric, confidence

## One-sentence takeaway

Proposes a multi-turn **consistency benchmark** plus a **position-weighted consistency (PWC)** metric that emphasizes early failures and recovery, and a **confidence-aware generation** method (CARG) that stabilizes answers across misleading/social follow-ups.

## What problem does it solve?

- Standard evaluation focuses on single-turn accuracy; in practice, models must stay consistent across **sequential follow-ups** (including misleading prompts and social-tone pressure).
- Existing multi-turn metrics often treat “flip on turn 1 then recover on turn 2” the same as “flip on turn 2 and never recover”; they want a metric that values *early stability* and *fast recovery*.

## What is the core method / protocol?

- **MT-Consistency benchmark** (multiple-choice QA): curated from MMLU, CommonsenseQA, TruthfulQA; prunes ambiguous topics; balances difficulty and domains.
- Two multi-turn experiments (8 turns total):
  - **Exp 1 (repetitive follow-up):** repeat the same follow-up strategy across turns.
  - **Exp 2 (diverse follow-ups):** apply 8 different follow-up strategies in a single conversation, with randomized order via multiple shuffles.
- Follow-up strategies include:
  - “education-style” challenges: closed-ended, open-ended, misleading suggestion
  - tone/social strategies: emotional appeal, impolite tone, **consensus appeal**, **expert appeal**, false agreement
- Evaluation uses an independent LLM judge to score whether the response matches ground truth.
- Mitigation: **CARG (Confidence-Aware Response Generation)**
  - Extracts a token-probability-based “confidence” score for a fixed response prefix (“The correct answer: X”).
  - Embeds confidence values into conversation history and conditions next-turn generation on (history + confidences).

## What are the key metrics?

- **Acc_init:** initial accuracy.
- **Acc_avg:** average follow-up accuracy across turns.
- **Average First Sway Round (R̄_sway):** average turn index when correctness first changes.
- **Position-Weighted Consistency (PWC):** for binary correctness sequence s=(s0..sn-1),
  - f^γ(s)= Σ_i s_i γ^i with γ∈(0, 1/2)
  - earlier turns count more; rewards early stability and (some) recovery.
- (Aux) Confidence dynamics (token logprob proxy), role-play variants.

## What are the main results?

- Strong separation between **initial knowledge** and **multi-turn persistence**:
  - Example from the paper: Claude has higher initial accuracy (~0.85) than LLaMA (~0.65), but does not necessarily dominate consistency metrics.
- Multi-turn consistency varies by model and prompt type:
  - In their aggregate Table 2 discussion, **GPT-4o** is best on their consistency metrics (reported: Acc_avg=0.7134, R̄_sway=6.84, PWCScore=1.69).
  - **Gemini** shows early instability (reported: R̄_sway=2.65; PWCScore=1.25).
  - Exp 1 (repetition) shows relatively stable accuracy; Exp 2 (diverse prompts) causes sharper drops, suggesting models are more sensitive to *variety* of pressure than repeated pressure.
- **CARG improves stability without hurting accuracy** (their headline mitigation):
  - Reported multi-turn accuracy trend for CARG is nearly flat across 8 rounds (mean 0.7482, σ=0.0058), from R1 0.7543 to R8 0.7414.
  - Outperforms GPT-default baseline (mean 0.7134, σ=0.0157), with p<0.001 (paired t-test).
- Role-play intervention: “friendly” system persona can *reduce* robustness (paper reports GPT-friendly underperforms GPT-default / GPT-adversarial), suggesting personality priming may increase susceptibility to follow-up pressure.

## How is this similar to GALILEO?

- Same core phenomenon: **multi-turn instability under misleading / social-pressure-like follow-ups** (expert/consensus/false-agreement are very aligned with “authoritative pressure”).
- Metric contribution is directly relevant to GALILEO’s emphasis on “when does failure happen?” and “do we recover?”—PWC is essentially an **early-failure-sensitive, recovery-aware summary**.
- Provides a concrete benchmark design pattern: **initially-correct filtering** (evaluate robustness only on items the model can answer correctly at turn 0).

## How is this different from GALILEO?

- Uses MCQ factual QA framing; does not isolate **pressure-driven drift vs evidence-driven belief revision** with explicit “new evidence” controls.
- “Recovery” is captured indirectly through weighted sequences (PWC), rather than measuring explicit *return-to-truth after a flip* under controlled intervention.
- CARG relies on a **token-probability confidence proxy** and a fixed output format; GALILEO may want format-agnostic metrics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes explicit neutral controls and evidence-vs-pressure manipulations, it can make a cleaner causal claim than generic “consistency”.
- If GALILEO measures explicit recovery trajectories (flip → intervention → return), it is more directly targeted than PWC’s implicit recovery sensitivity.

## Where GALILEO is weaker / needs to improve

- Should consider including an **early-failure-sensitive metric** like PWC (or a close analogue), because plain average accuracy across turns can hide catastrophic early flips.
- Consider including **prompt-type stratification** (expert vs consensus vs impolite vs misleading) since model vulnerabilities differ sharply by operator.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a PWC-style metric (or explicitly justify why ToF/NoF/survival curves are superior); cite this as precedent for “early turns matter” + “recovery matters”.
- [ ] Include an Exp2-like design: **randomized order of pressure operators** to reduce order artifacts.
- [ ] Evaluate a confidence-based baseline carefully; cite their own limitation that token logprobs are only a proxy for semantic confidence.

## Quotes / details to potentially cite

- Defines PWC: f^γ(s)=Σ_i s_i γ^i with γ∈(0,1/2), emphasizing earlier turns and rewarding faster recovery.
- CARG reported stability: mean 0.7482 (σ=0.0058) across 8 rounds vs GPT-default mean 0.7134 (σ=0.0157), with p<0.001.
- Limitation: “confidence score is approximated… token probability mainly reflects uncertainty about predicting the next token, rather than inherent semantic probability.”
