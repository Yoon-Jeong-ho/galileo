# Rapid Review — Top 10 shortlist (rolling)

This is a rolling shortlist of the **10 most important adjacent papers**.

For each item, we maintain:
- What it contributes
- What it misses relative to GALILEO
- What we should borrow
- How we should change GALILEO (experiments/methods/claims)

> NOTE: This list is expected to change frequently until ~30–50 papers are read.

## Shortlist

1) **Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution** (Zhang, arXiv 2026)
   - Contributes: a clear mechanistic claim about compliance under knowledge conflict: failures can be driven by **angular rotation** ("orthogonal interference") rather than **magnitude/norm dilution**, evaluated across multiple model families.
   - Misses vs GALILEO: primarily diagnostic/representational; does not (from rapid read) provide a complete behavioral protocol for long-horizon pressure/recovery, nor a practical intervention that works without internal access.
   - Borrow: the **radial vs angular** decomposition as an explanation tool; a reviewer-friendly counterpoint to “confidence/norm drops” as a universal mechanism.
   - How to change GALILEO: add a subsection/ablations showing when drift happens with **stable norms but changing direction**; avoid over-relying on scalar confidence proxies.

2) **Large Language Models Are More Persuasive Than Incentivized Human Persuaders** (Schoenegger et al., arXiv 2025)
   - Contributes: a direct head-to-head comparison of **directional persuasion compliance** in an interactive quiz chat, showing a frontier LLM persuader can outperform **incentivized human persuaders** in both truthful (toward correct answers) and deceptive (toward incorrect answers) steering.
   - Misses vs GALILEO: the outcome is (largely) single-episode compliance/accuracy rather than long-horizon drift + recovery; limited diagnostic separation of evidence-driven revision vs pressure-driven drift.
   - Borrow: (i) a clean *directional persuasion* task definition, (ii) outcome coupling to user utility (accuracy/earnings), (iii) human-baseline framing for persuasion strength.
   - How to change GALILEO: add a persuasion-strength evaluation axis (truthful + deceptive) with compliance + utility metrics; consider a human-baseline or strong scripted baseline for effect-size context.

3) **Improving Interactive In-Context Learning from Natural Language Feedback** (Cook et al., arXiv 2026)
   - Contributes: a scalable recipe for turning single-turn verifiable tasks into **multi-turn didactic interactions** (teacher critiques, student revises) and using RL to train the student’s ability to *actually update* based on natural-language corrective feedback.
   - Misses vs GALILEO: training signal is primarily end-task correctness; does not directly separate evidence-driven revision vs pressure-driven drift unless the verifier/teacher is designed to do so.
   - Borrow: (i) the **information-asymmetry teacher** idea (same base model but privileged info), (ii) multi-turn RL framing for “interactive plasticity”, (iii) OOD transfer story (train on math interactions → gains in coding/puzzles/mazes).
   - How to change GALILEO: add an explicit related-work + baseline slice: single-turn RL vs multi-turn didactic RL; include an “acknowledge-but-don’t-rewrite” failure probe and report multi-turn success curves.

4) **Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models** (Pi et al., arXiv 2025)
   - Contributes: identifies and quantifies a distinct multimodal failure mode: **visual sycophancy** is substantially worse than text sycophancy for “equivalent” information, termed the **sycophantic modality gap**; proposes **Sycophantic Reflective Tuning (SRT)** to reduce compliance to *misleading* user opinions without collapsing the ability to accept *corrective* feedback.
   - Misses vs GALILEO: evaluation is largely MME-style binary VQA with templated opinion injection; less coverage of longer-horizon drift/recovery and of open-ended multimodal tasks.
   - Borrow: (i) the **text-vs-vision gap** framing (compare image to “gold text description”), (ii) the correction-vs-sycophancy trade-off metrics, (iii) the staged **textualize → reflect (misleading vs corrective) → conclude** design.
   - How to change GALILEO: add a multimodal sycophancy slice and report a modality gap; include a “visual confidence sweep” (resolution/noise/blur) to test whether pressure failures track perceptual uncertainty.

5) **Measuring Sycophancy of Language Models in Multi-turn Dialogues** (Hong et al., Findings of EMNLP 2025)
   - Contributes: concrete multi-turn social-pressure protocol + two extremely usable metrics: **Turn of Flip (ToF)** (time-to-failure proxy) and **Number of Flips (NoF)** (instability/oscillation).
   - Misses vs GALILEO: does not explicitly separate *evidence-driven belief revision* from *pressure-driven drift*; limited focus on *recovery after flip*.
   - Borrow: ToF/NoF framing and third-person prompting baseline (reported up to 63.8% reduction in debate).
   - How to change GALILEO: map our metrics onto a ToF/NoF-style vocabulary (or add direct analogues) and highlight our additional controls + recovery measurements.

6) **ELEPHANT: Measuring and understanding social sycophancy in LLMs** (Cheng et al., arXiv 2025)
   - Contributes: a theory-grounded expansion from “explicit agreement” → **social sycophancy as face-preservation** (validation, indirectness, framing, moral inconsistency), plus an evaluation showing LLMs preserve user face far more than humans in advice/wrongdoing contexts.
   - Misses vs GALILEO: does not center multi-turn *time-to-failure* / recovery dynamics; relies on human baselines and social-dimension classifiers rather than drift-vs-revision controls.
   - Borrow: the face-preservation taxonomy; the *stance-swap* moral-consistency probe (user adopts either side) as a clean way to reveal side-dependent affirmation.
   - How to change GALILEO: add “face channels” of pressure as first-class metrics/labels (validation/framing adoption), not only explicit belief agreement; include a moral-consistency or stance-swap subtest.

7) **Sycophancy Hides Linearly in the Attention Heads** (Genadi et al., arXiv 2026)
   - Contributes: a mechanistic localization claim that **correct→incorrect sycophancy** is most *steerable* via a **sparse subset of middle-layer attention heads**, plus evidence those heads attend disproportionately to **user doubt** cues.
   - Misses vs GALILEO: intervention relies on **internal activation access** + per-model head selection; less emphasis on long-horizon survival/recovery metrics.
   - Borrow: (i) the “**separable vs steerable**” distinction, (ii) head-level sparsity as a hypothesis about where social-pressure cues route.
   - How to change GALILEO: add an interpretability appendix: test whether pressure-driven flips are linearly decodable and whether a small set of attention heads dominates.

8) **T3: Benchmarking Sycophancy and Skepticism in Causal Judgment** (Chang, arXiv 2026)
   - Contributes: a clean **sensitivity vs specificity** decomposition (Utility vs Safety; Sheep vs Wolves), plus explicit measurement of **calibrated abstention** and **multi-turn flip dynamics** (Good Flip vs Bad Flip) under social + epistemic pressure.
   - Misses vs GALILEO: centered on causal-judgment vignettes rather than generic belief drift/revision; limited focus on recovery trajectories.
   - Borrow: Utility/Safety reporting; GoodFlip/BadFlip asymmetry.
   - How to change GALILEO: add an explicit endorse-true vs reject-false breakdown and track good/bad flip asymmetry.

9) **Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment** (Laban et al., arXiv 2023; v2 2025)
   - Contributes: a minimal, highly reusable 2–3 turn **FlipFlop** protocol (challenge: “Are you sure?”) with clean summary metrics: **flip rates** (Any/Correct/Wrong→Flip) and the **FlipFlop effect** ΔFF (final−initial accuracy).
   - Misses vs GALILEO: does not cleanly separate **evidence-driven revision** from **pressure-driven drift** (challenge is confirmatory); limited treatment of longer-horizon recovery dynamics.
   - Borrow: ΔFF + Correct/Wrong→Flip as reviewer-friendly, task-agnostic reporting; challenger-utterance variants (authority/persona) as sensitivity probes.
   - How to change GALILEO: add a FlipFlop-style baseline slice and report correct-vs-wrong flip asymmetry to distinguish helpful self-correction from harmful sycophantic drift.

10) **AREG: Adversarial Resource Extraction Game for Evaluating Persuasion and Resistance in Large Language Models** (Sakhawat & Sadab, arXiv 2026)
   - Contributes: an outcome-based *interactive* persuasion benchmark as a **zero-sum negotiation** over a $100 endowment, with a deterministic **LLM arbiter** that counts only new unconditional transfers; reports **dual Elo ratings** for persuasion (offense) and resistance (defense).
   - Misses vs GALILEO: operationalizes social influence narrowly as monetary commitment extraction; relies on judge parsing of commitments; does not directly probe evidence-vs-pressure belief revision except indirectly via dialogue strategy.
   - Borrow: (i) explicit **attack vs defend** role separation and reporting (C-Elo vs V-Elo), (ii) evidence that the two capabilities are **weakly correlated** (rho ~ 0.33), (iii) strategy analysis: incremental commitment-seeking vs verification-seeking defenses.
   - How to change GALILEO: add role-asymmetric metrics or a split “offense/defense” reporting table; consider a controlled stateful game slice where outcomes are verifiable and incremental (not just binary success).

## Changelog

- 2026-02-19: Added *Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models* (Pi et al., arXiv 2025) as a key multimodal sycophancy neighbor (sycophantic modality gap; SRT to reduce misleading-opinion compliance without making models overly stubborn). Displaced *Large language models can effectively convince people to believe conspiracies* (Costello et al., arXiv 2026): important downstream belief-change evidence, but less directly diagnostic/mitigative for multimodal sycophancy.
- 2026-02-19: Added *AREG: Adversarial Resource Extraction Game for Evaluating Persuasion and Resistance in Large Language Models* (Sakhawat & Sadab, arXiv 2026) as a key interactive persuasion/resistance neighbor (zero-sum negotiation; dual Elo for offense vs defense; rho ~ 0.33 dissociation). Displaced *Emergent Persuasion: Will LLMs Persuade Without Being Prompted?* (Chang et al., AAAI 2026 AIGOV Workshop; arXiv 2025): useful non-misuse framing, but narrower (mostly first-turn attempt rates) than AREG’s explicit multi-turn offense/defense dissociation.
- 2026-02-19: Added *Improving Interactive In-Context Learning from Natural Language Feedback* (Cook et al., arXiv 2026) as a key training-method neighbor for **multi-turn critique integration / interactive plasticity** (didactic dialogues + information-asymmetry teacher + RL). Displaced *TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series* (Qin et al., arXiv 2025): strong SITS foundation work, but less central than interactive feedback-learning to GALILEO’s current multi-turn robustness narrative.
- 2026-02-19: Added *Large Language Models Are More Persuasive Than Incentivized Human Persuaders* (Schoenegger et al., arXiv 2025) as a key persuasion-capability neighbor (LLM vs incentivized humans; truthful + deceptive steering; compliance + accuracy/earnings). Displaced *Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing* (Hong et al., AAAI 2026; arXiv 2025): strong EO foundation work, but off-theme relative to GALILEO’s multi-turn social pressure / persuasion focus.
- 2026-02-19: Added *Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution* (Zhang, arXiv 2026) as a key mechanistic neighbor for compliance/sycophancy under knowledge conflict (angular rotation / orthogonal interference vs norm dilution). Displaced *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities* (Astruc et al., arXiv 2024/2025): strong EO foundation work, but less central to the current compliance/sycophancy mechanism emphasis.
- 2026-02-19: Added *Large language models can effectively convince people to believe conspiracies* (Costello et al., arXiv 2026) as a key multi-turn persuasion + belief-change neighbor (bunk vs debunk; guardrails not sufficient; corrective conversation recovery). Displaced *FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing* (Yang et al., arXiv 2025): valuable geospatial multimodal FM work, but less central to GALILEO’s multi-turn robustness / persuasion theme.
- 2026-02-19: Added *Emergent Persuasion: Will LLMs Persuade Without Being Prompted?* (Chang et al., AAAI 2026 AIGOV Workshop; arXiv 2025) as a key non-misuse persuasion-safety neighbor (UnPromptedAPE; benign SFT → harmful unprompted persuasion). Displaced *Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks* (Li, Krishnan, Padman, arXiv 2025): useful time-to-event framing, but less central than emergent persuasion for the current shortlist.
- 2026-02-19: Added *FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing* (Yang et al., arXiv 2025) as a key SAR multimodal foundation-model neighbor (geo-metadata + structured text + SCIO denoising). Displaced *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models* (Lu et al., arXiv 2026): interesting mechanistic persona-stability work, but less central than EO/SAR multimodal foundations for the current shortlist.
- 2026-02-19: Added *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models* (Lu et al., arXiv 2026) as a mechanistic persona-stability neighbor (Assistant Axis + activation capping; links drift in therapy/meta-reflection to unsafe/bizarre outputs; mitigates persona-based jailbreaks). Displaced *Modeling and Predicting Multi-Turn Answer Instability in Large Language Models* (He et al., arXiv 2025): useful Markov/stationary framing, but less central than persona-drift stabilization for our current narrative.
- 2026-02-19: Added *Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment* (Laban et al., arXiv 2023) as a key multi-turn **challenge→flip** sycophancy protocol (ΔFF + flip-rate metrics). Displaced *CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse* (Geng et al., arXiv 2026): useful benchmark infrastructure, but less central than a foundational FlipFlop-style protocol.
- 2026-02-19: Added *TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series* (arXiv 2025) as a key **SITS** neighbor (hierarchical backbone + structured spatiotemporal attention; MillionST with 10 timestamps over 5 years). Displaced *Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD* (Tan et al., EMNLP 2025): useful dual-axis persuasion framing, but less central if the related-work emphasis is shifting toward EO foundation models.
- 2026-02-19: Added *Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing* (AAAI 2026; arXiv 2025) as a key **EO foundation model** neighbor focused on missing-band + cross-sensor + cross-resolution robustness. Displaced *Consistency of Large Reasoning Models Under Multi-Turn Attack* (Li, Krishnan, Padman, arXiv 2026): useful trajectory taxonomy, but less central if the related-work emphasis is shifting toward EO foundation models.
