# Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion

- Year: 2025 (arXiv; accepted AAAI 2026)
- Venue: arXiv (AAAI 2026 acceptance noted)
- Authors: Chen Han, Yijia Ma, Jin Tan, Wenzhen Zheng, Xijin Tang
- URL: https://arxiv.org/abs/2511.07267
- BibTeX key (if we add it): han2025beyond
- Tags: misinformation, multi-agent, debate, evidence-retrieval, persuasion, human-study

## One-sentence takeaway

ED2D adds evidence retrieval to a structured multi-agent debate fact-checker and shows that when the system’s prediction is correct its debate-style explanations can be as persuasive as expert fact-checks, but when it’s wrong the same persuasive transcripts can backfire and reinforce misinformation.

## What problem does it solve?

- Multi-agent debate (MAD) improves misinformation detection, but prior work mostly optimizes *classification accuracy* and under-emphasizes (a) grounding in verifiable evidence and (b) whether the generated debate/explanations actually *change human beliefs / sharing intent*.
- Need to understand the *benefits and risks* of deploying persuasive, explanation-generating debate systems for real-world misinformation intervention.

## What is the core method / protocol?

- ED2D: Evidence-based Debate-to-Detect.
- Architecture: 5-stage structured debate (Opening, Rebuttal, Free Debate, Closing, Judgment) with two teams (Affirmative vs Negative) and judge agents scoring across multiple dimensions.
- Key extension vs prior D2D: an evidence retrieval module used during Free Debate (and referenced in Judgment).
  - Extract up to ~5 salient entities/relations from the claim.
  - Retrieve evidence snippets via a Wikipedia-based API.
  - Classify stance of each snippet toward the claim (support/refute/neutral) with an LLM.
  - Inject evidence into debater turns; preserve neutral evidence for judges.
- Evaluation has two parts:
  - Detection accuracy on 3 datasets (Weibo21, FakeNewsDataset, and their new Snopes25).
  - Human-subject persuasion study (200 participants) comparing conditions: Control (no explanation), ED2D explanation, Snopes expert explanation, Combined.

## What are the key metrics?

- Detection: Acc / Precision / Recall / F1 (reported across datasets).
- Persuasion (per-claim, 7-point Likert):
  - Belief in the claim
  - Willingness to share
  - Emotional agreement
- Also report human veracity-judgment accuracy under different explanation conditions; plus a post-test (no explanations) to see if exposure improves future independent detection.

## What are the main results?

- ED2D achieves best detection performance across all three benchmarks among compared baselines (BERT/RoBERTa, single-LLM prompting baselines, and debate baselines), and adding evidence generally helps all LLM-based methods.
- Persuasion study:
  - When ED2D’s label is *correct*, its debate transcripts are about as persuasive as Snopes expert write-ups for correcting belief and reducing sharing intent for false claims.
  - When ED2D *misclassifies*, its explanations can *mislead*: it may increase belief/sharing for false claims labeled true (and reduce belief in true claims labeled false), and can partially counteract Snopes even in a Combined condition.
- Suggests a core deployment risk: “persuasive explanation” + “wrong label” is worse than just being wrong.

## How is this similar to GALILEO?

- Directly about *persuasion / belief change* and how multi-turn, structured interactions (debate transcripts) affect user beliefs.
- Highlights a key safety/robustness theme: explanation systems can be *harmful when incorrect*, especially if they are rhetorically convincing.
- Uses multi-agent protocol + judges + scoring dimensions: similar design space to agentic evaluation / oversight setups.

## How is this different from GALILEO?

- Focused on misinformation fact-checking and human persuasion outcomes, not primarily on assistant “sycophancy” or instruction-following robustness.
- Evidence retrieval is Wikipedia/API-centric and integrated into debate; GALILEO may be targeting different grounding sources, threat models, or interaction protocols.
- Their risk finding is tied to *prediction correctness* on a claim; GALILEO may care more about robustness to adversarial user pressure, multi-turn drift, or goal/stability metrics.

## Where GALILEO is stronger / cleaner (if true)

- Opportunity: GALILEO can explicitly separate (1) correctness and (2) persuasiveness/linguistic confidence to avoid “confidently persuasive when wrong” failure modes, and evaluate this under adversarial pressure.
- Opportunity: more systematic *calibration / abstention* policies (e.g., refuse to persuade when uncertainty is high) than ED2D’s binary-real/fake output.

## Where GALILEO is weaker / needs to improve

- If GALILEO doesn’t yet include human-subject endpoints, this paper is a reminder that “good explanations” should be tested for *behavioral outcomes* (belief + sharing) and not just model-side metrics.
- Need explicit evaluation of the “misleading persuasion when wrong” regime (the paper shows this matters even when expert explanations are shown alongside).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an eval slice: when the system is wrong, measure whether its explanation increases user belief/sharing versus control (i.e., *harmful persuasiveness*), and compare to safer baselines.
- [ ] Consider a design pattern: require *evidence-grounded citations + uncertainty reporting* before allowing persuasive framing.
- [ ] In related work, cite ED2D as evidence that multi-agent debate + evidence retrieval can be persuasive but introduces a “backfire when wrong” risk.

## Quotes / details to potentially cite

- ED2D evidence module steps: entity/relation extraction → retrieval → stance classification → evidence integration (during Free Debate).
- Snopes25 dataset: 448 claims with professional fact-check reports (Jan–Jun 2025), intended to be after GPT-4o cutoff to reduce leakage.
- Key risk statement (paraphrase): when ED2D misclassifies, explanations may reinforce misconceptions, even alongside accurate expert explanations.
