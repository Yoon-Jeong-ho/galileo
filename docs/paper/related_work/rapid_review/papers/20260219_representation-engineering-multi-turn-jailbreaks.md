# A Representation Engineering Perspective on the Effectiveness of Multi-Turn Jailbreaks

- Year: 2025
- Venue: arXiv (listed as ICML category on arXiv page)
- Authors: Mark Russinovich; Ahmed Salem; Santiago Zanella-Beguelin; Daniel Jones; Giorgio Severi; Eugenia Kim; Keegan Hines; Amanda Minnich; Yonatan Zunger; Ram Shankar Siva Kumar
- URL: https://arxiv.org/html/2507.02956v1
- BibTeX key (if we add it): russinovich2025repe-crescendo
- Tags: multi-turn, jailbreak, crescendo, representation-engineering, circuit-breakers, safety

## One-sentence takeaway

Multi-turn Crescendo jailbreaks can keep a model’s intermediate representations in a “benign/retain-like” region even while eliciting harmful outputs, explaining why single-turn RepE defenses like circuit breakers often fail to generalize to multi-turn attacks.

## What problem does it solve?

- Provides a mechanistic/representation-level explanation for why multi-turn jailbreaks (specifically Crescendo) bypass defenses that perform well on single-turn jailbreaks.
- Frames the issue as a *representation generalization gap*: defenses trained to separate benign vs harmful single-turn distributions do not reliably flag harmful behavior when the same harmful output tokens occur in a multi-turn conversational context.

## What is the core method / protocol?

- Target models: Llama-3-8B-Instruct and a circuit-breaker-hardened variant (Llama-3-8B-Instruct-RR).
- Generate multi-turn Crescendo jailbreak conversations (automated via PyRIT + LLM scorer; then manually verify a subset).
- Representation reading / probing pipeline:
  - Build single-turn “benign” vs “harmful” representation datasets using the *retain* dataset vs the *circuit breaker (CB)* dataset from Zou et al. (circuit breakers).
  - Extract intermediate hidden representations for response tokens (token-level vectors in R^4096) at a chosen layer (original Llama: last layer; CB model: the layer where circuit breakers were inserted).
  - Fit PCA on single-turn representations for visualization; project Crescendo final-response token representations into same PCA space.
  - Train an MLP probe to classify token representations as “harmful (CB-like)” vs “benign (retain-like)”; evaluate how the fraction classified harmful changes with number of turns k included in context.
- Key ablation-style analyses:
  - Vary k = number of recent turns included in context for the *same final response tokens*.
  - “Single Prompt” variant: put the entire conversation history into one user prompt (tests whether assistant-role tokens matter).
  - “Masked Responses”: mask earlier assistant responses from attending to the final response (tests influence of the model’s own earlier generations).
  - Replace history with the direct attack objective (single-turn harmful prompt).

## What are the key metrics?

- ASR (attack success rate) for Crescendo (via PyRIT + scorer LLM; plus manual verification for selected cases).
- Probe-based metric: percentage of final-response token representations classified as “harmful” (CB-like) by an MLP trained on single-turn retain vs CB representations.
- Qualitative/visual: PCA separation and shift of token clusters between “retain-like” vs “CB-like” regions.

## What are the main results?

- Crescendo conversation history shifts representations of the *final harmful response tokens* toward the benign/retain distribution relative to prompting with the direct attack objective.
- Increasing the number of turns included (k) generally decreases the fraction of tokens the probe labels as harmful; often a sharp drop from k=1 to k=2.
- Circuit-breaker-hardened model still shows substantial vulnerability to Crescendo (reported ~54% ASR for automated Crescendo on the circuit-breaker model in their setup), despite very low ASR reported for unseen single-turn attacks in prior circuit breaker work.
- “Single Prompt” (history as one user prompt) produces nearly identical representation effects, suggesting the model does not strongly privilege assistant-role tokens vs user-provided context containing similar content.
- Masking earlier assistant responses sometimes increases harmful classification, indicating the model’s own prior responses can help keep later representations benign—but effect is example-dependent.

## How is this similar to GALILEO?

- Both care about *multi-turn dynamics* and why multi-turn context changes model behavior relative to single-turn evaluation/defenses.
- Methodologically adjacent in spirit: using intermediate signals (here: representations + probes) to explain behavioral phenomena rather than only reporting end-task metrics.

## How is this different from GALILEO?

- This paper is about *safety/jailbreak robustness* and representation-space analyses; GALILEO focuses on multi-turn reliability/behavioral degradation patterns (not jailbreaks per se).
- They use supervised probes trained on retain-vs-harmful datasets, while GALILEO’s core contributions are evaluation methodology/metrics and empirical failure dynamics (as currently framed).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes model-agnostic evaluation and avoids probe interpretability assumptions, we can position GALILEO as more directly tied to observable task failure curves and less reliant on the semantic validity of representation probes.

## Where GALILEO is weaker / needs to improve

- If reviewers want mechanistic explanations for *why* multi-turn failure curves happen, this paper is a good example of a compact mechanistic story; GALILEO may benefit from a light-touch “explanatory analysis” section (even if not full mechanistic interpretability).

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work framing: explicitly cite multi-turn *generalization gaps* (defenses/constraints trained on single-turn inputs failing in multi-turn contexts) as an analogy to GALILEO’s multi-turn reliability gap.
- [ ] Add a short paragraph in related work contrasting:
  - (a) multi-turn jailbreaks that keep representations “benign-like” while eliciting harmful content vs
  - (b) multi-turn task interactions that keep surface form plausible while correctness degrades (GALILEO).
- [ ] Consider an optional appendix experiment: a simple probe-free diagnostic showing that the *same final answer tokens* can be judged differently by safety filters / classifiers depending on prior turns (analogy to representation shift). (Only if low-cost.)

## Quotes / details to potentially cite

- Abstract-level claim: safety-aligned LMs “often represent Crescendo responses as more benign than harmful, especially as the number of conversation turns increases,” motivating mitigations targeting the multi-turn generalization gap.
- Method insight: sharp drop in harmful-probe rate from k=1 to k=2 in many examples; overall decreasing harmful classification as more conversation turns are included.
- Finding: “Single Prompt” (history as one user prompt) yields nearly identical representation/probe behavior, suggesting role tokens are not the driver.
