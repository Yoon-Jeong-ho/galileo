# Multi-Turn Jailbreaks Are Simpler Than They Seem

- Slug: multi-turn-jailbreaks-simpler
- Year: 2025
- Venue: COLM 2025 SoLaR Workshop (arXiv)
- Authors: Xiaoxue Yang; Jaeha Lee; Anna-Katharina Dick; Jasper Timm; Fei Xie; Diogo Cruz
- Links:
  - paper: https://arxiv.org/abs/2508.07646
  - code (if any): https://github.com/diogo-cruz/multi_turn_simpler
- Bibtex: https://doi.org/10.48550/arXiv.2508.07646

## 1) What problem does it study?

Automated **multi-turn jailbreak** robustness: why multi-turn jailbreaks appear much more effective than single-turn jailbreaks, and whether that “multi-turn advantage” is genuinely about sophisticated dialogue dynamics or mostly about **having more chances + learning from refusals**.

## 2) Experimental setup (what is being measured?)

- Task(s): jailbreak (harmful instruction elicitation) on **StrongREJECT** harmful behaviors (30 behaviors; 5 per category).
- Perturbation/pressure type: adversarial prompting; primarily the **“Direct Request”** tactic (professional/authoritative framing), in single-turn and multi-turn versions.
- Multi-turn? Y
  - default max turns: *n_turns = 8*
  - refusal retries budget: *n_refusals = 10* (cumulative across an attack)
  - may reattempt full attacks: *n_attacks* (default 1)
- Metrics:
  - **StrongREJECT score** in [0,1] (rubric-based: non-refusal + specificity/convincingness)
  - analyses of score vs number of turns / number of attempts
  - evaluator reliability checks (human correlation; reported lower in multi-turn)
  - for “reasoning models”: score vs reasoning effort/tokens

## 3) Key findings (bullet)

- **Multi-turn “beats” naive single-turn**, but much of that gap is explained by **extra sampling opportunities**:
  - when single-turn attacks are given an **equivalent number of retries/attempts**, performance becomes *approximately equivalent* to multi-turn for the core tactic studied.
- **Refusal feedback matters**: letting the attacker see the refusal and retry can improve success relative to independent restarts (attacker learns what phrasing triggers refusals).
- **Public benchmark results may overestimate robustness** because many evaluations resemble “single attempt, no adaptive retry”, missing multi-turn / retry effects.
- **Evaluator brittleness increases in multi-turn**:
  - StrongREJECT evaluator correlation is lower in multi-turn than single-turn (they cite ~0.92 single-turn vs ~0.82 multi-turn for Direct Request), and some tactics had too many inaccuracies to rely on.
  - evaluation errors can compound with many attempts/turns.
- **Reasoning effort can increase attack success** for reasoning models (counterintuitive direction vs “more compute ⇒ more robust” narratives).
- **Attack success correlates within providers/model families**, suggesting newly released sibling models can be predictably vulnerable.

## 4) Limitations / threats

- Primary conclusions are drawn mainly from the **Direct Request** tactic; may not generalize to richer multi-turn tactics (e.g., Crescendo).
- Heavy reliance on an LLM-based evaluator; multi-turn evaluation appears less reliable, and errors may compound as attempts increase.
- Uses GPT-4o-mini as attacker and evaluator in the main setup; shared-model bias could inflate success rates.

## 5) How it relates to GALILEO

- What we can cite it for:
  - A concrete argument that **multi-turn vulnerability can be a “multi-sampling / retry budget” artifact**, not necessarily a uniquely multi-turn phenomenon.
  - A caution that robustness evaluations should specify the **attacker adaptivity and retry budget**, or they will systematically overestimate safety.
  - Evidence that **inference-time reasoning/effort can *worsen* safety outcomes** in some settings.
- Where we differ (our delta):
  - This is **jailbreak/harmful-content** oriented; GALILEO targets **social pressure / sycophancy / belief drift vs revision**, with explicit neutral controls and recovery measurement.
- Direct mapping:
  - Survival ↔ “score vs turns/attempts” suggests a time-to-event framing; we can treat turns/attempts as exposure time and compare policies under equal budgets.
  - TOF ↔ first turn where StrongREJECT score crosses a harmfulness threshold (conceptual analogue).
  - Recovery ↔ not a focus here (mostly first-success accounting).
  - Neutral Re-asking Control ↔ their “equivalent number of attempts” comparison is analogous to a control that holds *opportunity count* fixed.

## 6) Quote-able lines

- Paraphrase target: multi-turn jailbreak gains are “approximately equivalent” to **resampling single-turn attacks** when retry/feedback is accounted for.
- Paraphrase target: **more reasoning effort** can correlate with **higher jailbreak success**.

## 7) Actions

- [ ] Add to paper: related-work paragraph in robustness-evaluation section: “multi-turn vs retry-budget equivalence; resampling baseline; evaluator brittleness in multi-turn.”
- [ ] Add to bib
