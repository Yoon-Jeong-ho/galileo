# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models

- Slug: persuade-me-if-you-can
- Year: 2025
- Venue: NeurIPS Workshop on Multi-Turn Interactions in Large Language Models (MTI-LLM) (workshop paper; arXiv v3 Feb 2026)
- Authors: Nimet Beyza Bozdag; Shuhaib Mehri; Gokhan Tur; Dilek Hakkani-Tür
- Links:
  - paper: https://arxiv.org/abs/2503.01829
  - html: https://arxiv.org/html/2503.01829v3
  - code (if any): https://beyzabozdag.github.io/PMIYC/
- Bibtex: https://doi.org/10.48550/arXiv.2503.01829

## 1) What problem does it study?

How to **measure (at scale)** both:
- **Persuasive effectiveness**: how well an LLM can persuade another agent.
- **Susceptibility / resistance to persuasion**: how easily an LLM changes its stance under persuasive pressure.

Motivation includes both “persuasion for good” and safety risks (misinformation, coercion/jailbreak-like dynamics), especially in **multi-agent** settings.

## 2) Experimental setup (what is being measured?)

- Task(s): two-agent “Persuader (ER)” vs “Persuadee (EE)” conversation about a claim.
  - Persuader argues *for* the claim (even if it may not personally agree).
  - Persuadee produces natural-language replies and a **self-reported agreement score**.
- Perturbation/pressure type: conversational persuasion (argumentation / addressing counterpoints) in:
  - **Subjective claims** (debate-style opinions)
  - **Misinformation** (Persuader pushes an incorrect answer)
- Multi-turn? Y
  - Single-turn condition: total turns t = 3 (EE initial stance → ER argument → EE final decision)
  - Multi-turn condition: total turns t = 9 (multiple ER attempts); **early stop** if EE reaches “Completely Support (5)” after ER has spoken at least once.
- Measurement primitive: 5-point Likert scale for EE stance:
  - 1 Completely Oppose … 5 Completely Support.

### Data
- Subjective: 961 claims from Durmus et al. (2024) + Perspectrum (Chen et al., 2019).
- Misinformation: 817 TruthfulQA questions paired with a target **incorrect** answer; Persuader tries to convince EE to accept that incorrect answer.

### Metrics
- **Normalized Change in Agreement (NCA)**: normalized (s_t − s_0) with range [-1, 1], normalizing by distance to the boundary (5 − s_0) or (s_0 − 1) depending on direction.
  - Used both as:
    - Persuader “effectiveness” (avg NCA when model is ER)
    - Persuadee “susceptibility” (avg NCA when model is EE)
- Turn-level trajectory: average EE agreement score per EE turn (used for analysis of *when* persuasion happens; they observe strongest effect in first 1–2 persuasive attempts).
- Reliability checks:
  - EE initial opinion consistency across repeated queries (std dev, match rate, etc.).
  - Self-report validation via human annotation + action-based (MCQ) tracking.

## 3) Key findings (bullet)

- Framework: PMIYC enables **automated multi-turn persuasion** experiments across many model pairings without requiring full human evaluation.
- Effectiveness ranking (headline): **Llama-3.3-70B** and **GPT-4o** are similarly strong persuaders; both outperform **Claude 3 Haiku** by ~30% (as stated in abstract).
- Resistance is domain-dependent:
  - In **misinformation**, **GPT-4o** shows **>50% greater resistance** to persuasion than Llama-3.3-70B (abstract claim).
- Multi-turn generally increases persuasion (both effectiveness and susceptibility increase in multi-turn vs single-turn).
- Turn dynamics: persuasion gains **peak in the first 1–2 persuasive attempts** (later turns show diminishing returns).
- Important artifact / interaction effect:
  - They observe cases where a strong Persuadee “pulls” a weaker Persuader away from the target stance; EE may still self-report agreement mid-conversation but then drops agreement at the final decision when reminded of the original claim.
  - They explicitly note these turn-level dynamics can expose **sycophancy-like over-accommodation** (temporary alignment that disappears when re-asked for the final stance).
- Self-report validation:
  - Human annotation: 3-way (agree/neutral/disagree) match ~76%, Cohen’s κ ≈ 0.63; avg absolute diff ≈ 0.51 on the 1–5 scale (reported in §5.3.1).
  - Action-based tracking: generally high alignment between self-reports and action choices (>90% in most settings); in misinformation, they report substantial rates of “genuine persuasion” (selecting the target incorrect answer + final agreement ≥ 4) on the order of ~64–78% (as described in §5.3.2).

## 4) Limitations / threats

- Core outcome relies heavily on **self-reported agreement**; while validated, it remains a potential surface for prompt/format sensitivity.
- No explicit separation between:
  - **evidence-driven belief revision** vs
  - **pressure-driven drift / conformity**.
  (They treat persuasion broadly; misinformation setting is closer, but still mixes “argument quality” with “social pressure”.)
- Not framed as survival/time-to-failure; primary metric is end-of-dialogue (plus trajectories), so it doesn’t directly provide censored time-to-event metrics.
- Interaction artifact: persuader stance can drift during the dialogue due to the persuadee’s counter-arguments, complicating interpretation (they discuss this in analysis).

## 5) How it relates to GALILEO

- What we can cite it for:
  - A scalable **multi-turn, agent-to-agent persuasion** evaluation harness.
  - A simple, comparable persuasion effect metric (**NCA**) and the general finding that multi-turn increases susceptibility.
  - Evidence that **turn-level stance trajectories** reveal patterns like temporary agreement that collapses on re-asking (useful precedent for GALILEO’s controls).
- Where we differ (our delta):
  - GALILEO focuses on **robustness under social pressure** with explicit **controls** (drift vs revision) and **recovery after flip**, rather than a general persuasion scoreboard.
  - GALILEO can report **time-to-failure / turn-of-flip** and recovery trajectories explicitly (PMIYC is closer to end-of-dialogue change + qualitative trajectory plots).
- Direct mapping:
  - Survival ↔ PMIYC turn-level agreement curves (but not a censored survival model).
  - TOF ↔ “first turn where agreement crosses threshold” could be derived from their per-turn agreement history.
  - Recovery ↔ their observation of final-step agreement drop after reminder is a proto-signal of “recovery-on-reasking”, but not formalized.
  - Neutral Re-asking Control ↔ their “final decision step with reminder of the claim” functions similarly and is worth citing as a sanity-check step.

## 6) Quote-able lines

- Abstract-level paraphrase targets:
  - PMIYC is an “automated framework for evaluating persuasiveness and susceptibility to persuasion in multi-agent interactions”.
  - “GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B.”
- Analysis paraphrase targets:
  - Most persuasion gains occur in first 1–2 attempts.
  - Final re-asking/reminder can reduce apparent agreement, suggesting mid-dialogue accommodation.

## 7) Actions

- [ ] Add to paper: related work section on automated multi-turn persuasion / susceptibility evaluation; emphasize “final decision reminder” as a control-like step.
- [ ] Add to bib
