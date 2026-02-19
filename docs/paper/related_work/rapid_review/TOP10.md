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

2) **Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing** (Hong et al., AAAI 2026; arXiv 2025)
   - Contributes: an optical RSFM explicitly designed to handle **arbitrary band compositions** and **wide resolution ranges** via (i) **per-band tokenization + band identity embeddings** and (ii) **resolution-adaptive patch embedding** (multi-kernel bank + pseudo-inverse resize).
   - Misses vs GALILEO: focused on optical multispectral (not a general multi-modal geospatial model); unclear from rapid read whether it supports temporal modeling beyond standard pretraining augmentations.
   - Borrow: (i) missing-band and cross-sensor evaluation slices as first-class claims, (ii) explicit band-identity encoding baseline, (iii) multi-scale semantic alignment framing.
   - How to change GALILEO: add explicit **missing-band / cross-sensor / cross-resolution** stress tests and (if needed) a tokenizer/pathway that preserves band identity while still allowing cross-band interaction.

3) **TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series** (Qin et al., arXiv 2025)
   - Contributes: a **hierarchical** SITS foundation model with a custom attention pattern (**spatiotemporal gyroscope attention**) aimed at capturing multiscale aligned space-time structure; plus a temporally richer pretraining set (**MillionST**, 10 timestamps over 5 years at 100k locations).
   - Misses vs GALILEO: appears specialized to *aligned* SITS (Sentinel-2) and masked modeling; unclear how it extends to heterogeneous sensors/modalities or misaligned/missing time series.
   - Borrow: (i) “temporal cardinality matters” narrative and dataset design, (ii) structured attention that respects SITS alignment, (iii) hierarchical/pyramid backbone as a strong default for multiscale geospatial phenomena.
   - How to change GALILEO: add at least one baseline/ablation contrasting **flat ViT vs hierarchical** backbones for time series, and explicitly report transfer vs number of timestamps.

4) **Large language models can effectively convince people to believe conspiracies** (Costello et al., arXiv 2026)
   - Contributes: strong evidence (3 pre-registered experiments; N=2,724) that **multi-turn LLM conversations can increase misbeliefs** ("bunking") about as effectively as they can decrease them ("debunking"), and that **standard guardrails may not prevent** harmful persuasion.
   - Misses vs GALILEO: focuses on downstream *human belief change* rather than a diagnostic decomposition of **when belief revision is justified vs pressure-driven drift**; interventions are mainly prompt-level ("only use accurate information") and post-hoc corrective conversation.
   - Borrow: (i) framing that drift/pressure can have **real-world belief/trust impacts**, (ii) a clean experimental contrast of **bunk vs debunk** as a multi-turn steering pressure, (iii) “corrective conversation” as a recovery concept.
   - How to change GALILEO: explicitly connect GALILEO stability metrics to downstream harm; add an evaluation slice measuring recovery after a harmful-steering attempt (post-correction) and test an “accuracy-only” constraint as a baseline stabilizer.

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

10) **Emergent Persuasion: Will LLMs Persuade Without Being Prompted?** (Chang et al., AAAI 2026 AIGOV Workshop; arXiv 2025)
   - Contributes: a concrete **non-misuse** threat model for persuasion (“unprompted persuasion”), plus an evaluation variant (**UnPromptedAPE**) and evidence that **SFT on benign persuasion data can generalize into harmful-topic persuasion attempts**.
   - Misses vs GALILEO: mostly first-turn attempt rates (less about long-horizon multi-turn dynamics / recovery); uses a specific base model family (Qwen2.5-7B) rather than broad coverage.
   - Borrow: UnPromptedAPE-style prompt removal as a clean protocol; “benign post-training → harmful unprompted persuasion” as a crisp governance-relevant claim.
   - How to change GALILEO: add an explicit unprompted-persuasion slice (remove persuasion instructions) and, if discussing post-training risks, cite benign persuasion fine-tuning as a mechanism for emergent harmful persuasion.

## Changelog

- 2026-02-19: Added *Simulated Adoption: Decoupling Magnitude and Direction in LLM In-Context Conflict Resolution* (Zhang, arXiv 2026) as a key mechanistic neighbor for compliance/sycophancy under knowledge conflict (angular rotation / orthogonal interference vs norm dilution). Displaced *AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities* (Astruc et al., arXiv 2024/2025): strong EO foundation work, but less central to the current compliance/sycophancy mechanism emphasis.
- 2026-02-19: Added *Large language models can effectively convince people to believe conspiracies* (Costello et al., arXiv 2026) as a key multi-turn persuasion + belief-change neighbor (bunk vs debunk; guardrails not sufficient; corrective conversation recovery). Displaced *FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing* (Yang et al., arXiv 2025): valuable geospatial multimodal FM work, but less central to GALILEO’s multi-turn robustness / persuasion theme.
- 2026-02-19: Added *Emergent Persuasion: Will LLMs Persuade Without Being Prompted?* (Chang et al., AAAI 2026 AIGOV Workshop; arXiv 2025) as a key non-misuse persuasion-safety neighbor (UnPromptedAPE; benign SFT → harmful unprompted persuasion). Displaced *Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks* (Li, Krishnan, Padman, arXiv 2025): useful time-to-event framing, but less central than emergent persuasion for the current shortlist.
- 2026-02-19: Added *FUSAR-KLIP: Towards Multimodal Foundation Models for Remote Sensing* (Yang et al., arXiv 2025) as a key SAR multimodal foundation-model neighbor (geo-metadata + structured text + SCIO denoising). Displaced *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models* (Lu et al., arXiv 2026): interesting mechanistic persona-stability work, but less central than EO/SAR multimodal foundations for the current shortlist.
- 2026-02-19: Added *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models* (Lu et al., arXiv 2026) as a mechanistic persona-stability neighbor (Assistant Axis + activation capping; links drift in therapy/meta-reflection to unsafe/bizarre outputs; mitigates persona-based jailbreaks). Displaced *Modeling and Predicting Multi-Turn Answer Instability in Large Language Models* (He et al., arXiv 2025): useful Markov/stationary framing, but less central than persona-drift stabilization for our current narrative.
- 2026-02-19: Added *Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment* (Laban et al., arXiv 2023) as a key multi-turn **challenge→flip** sycophancy protocol (ΔFF + flip-rate metrics). Displaced *CausalT5K: Diagnosing and Informing Refusal for Trustworthy Causal Reasoning of Skepticism, Sycophancy, Detection-Correction, and Rung Collapse* (Geng et al., arXiv 2026): useful benchmark infrastructure, but less central than a foundational FlipFlop-style protocol.
- 2026-02-19: Added *TiMo: Spatiotemporal Foundation Model for Satellite Image Time Series* (arXiv 2025) as a key **SITS** neighbor (hierarchical backbone + structured spatiotemporal attention; MillionST with 10 timestamps over 5 years). Displaced *Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD* (Tan et al., EMNLP 2025): useful dual-axis persuasion framing, but less central if the related-work emphasis is shifting toward EO foundation models.
- 2026-02-19: Added *Any-Optical-Model: A Universal Foundation Model for Optical Remote Sensing* (AAAI 2026; arXiv 2025) as a key **EO foundation model** neighbor focused on missing-band + cross-sensor + cross-resolution robustness. Displaced *Consistency of Large Reasoning Models Under Multi-Turn Attack* (Li, Krishnan, Padman, arXiv 2026): useful trajectory taxonomy, but less central if the related-work emphasis is shifting toward EO foundation models.
