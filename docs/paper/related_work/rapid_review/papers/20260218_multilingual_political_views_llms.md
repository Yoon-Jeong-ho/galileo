# Multilingual Political Views of Large Language Models: Identification and Steering

- Year: 2025
- Venue: arXiv (cs.CL) preprint
- Authors: Daniil Gurgurov; Katharina Trinley; Ivan Vykopal; Josef van Genabith; Simon Ostermann; Roberto Zamparelli
- URL: https://arxiv.org/abs/2507.22623
- BibTeX key (if we add it): gurgurov2025multilingual
- Tags: political-stance, multilingual, steering, robustness, activation-interventions

## One-sentence takeaway

Across 7 instruction-tuned open LLMs and 14 languages, political-compass orientations vary systematically with scale/language and can be steered reliably via simple activation “center-of-mass” interventions.

## What problem does it solve?

- Quantify whether modern open-source instruction-tuned LLMs have consistent political orientations (and how this varies by model size/family and language).
- Test whether such political orientations are *manipulable/controllable* at inference time rather than being a fixed artifact.

## What is the core method / protocol?

- Measurement:
  - Use the Political Compass Test (PCT) as an operationalization of political orientation (2D: economic left–right + authoritarian–libertarian).
  - Evaluate **7** instruction-tuned models (examples named: LLaMA-3.1, Qwen-3, Aya-Expanse) across **14 languages**.
  - For robustness, use **11 semantically equivalent paraphrases per statement** (in the target language) to reduce prompt phrasing brittleness.
- Steering:
  - Apply an inference-time activation intervention described as a **center-of-mass** / mean-activation-difference direction (citing Marks & Tegmark-style approach).
  - Demonstrate steering in multiple languages (named in intro: English, Turkish, Romanian, Slovenian, French), shifting outputs toward alternative ideological quadrants.

## What are the key metrics?

- PCT score coordinates / quadrant placement (economic axis, social axis) aggregated over items.
- Robustness across paraphrases (implicitly: variance / stability of PCT score under paraphrase perturbations).
- Steering effectiveness: shift in PCT coordinates after applying the activation intervention (directional movement toward target ideological region).

## What are the main results?

- **Scale effect:** larger models “consistently shift toward libertarian-left” (as reported; example figure compares Aya-Expanse 8B vs 32B).
- **Language effects:** substantial variation in measured orientation across languages and model families (i.e., bias is not monolithic).
- **Controllability:** a simple activation-direction intervention can reliably steer PCT-measured ideology across multiple languages.

## How is this similar to GALILEO?

- Treats “stance/orientation” as a measurable latent property of model behavior and studies how it changes under perturbations.
- Emphasizes *robust measurement* (paraphrase sets) rather than single-prompt conclusions.
- Demonstrates an explicit *intervention* mechanism (steering) analogous to “control knobs” GALILEO might consider for stabilizing/redirecting behavior.

## How is this different from GALILEO?

- Focus is political ideology (PCT) rather than truthfulness, epistemic stability, or evidence-driven belief revision (depending on GALILEO’s target).
- Uses a survey-style, single-turn test battery; less about multi-turn dynamics, persistence, or time-to-failure.
- Steering is via internal activation manipulation rather than prompt-based controls / training-time methods / conversational interventions.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets *stability over time / across turns*, it can provide a clearer causal story about drift vs revision than a static PCT battery.
- If GALILEO uses task-grounded labels (truth conditions, evidence), it may avoid the normative/construct-validity issues of mapping to ideological axes.

## Where GALILEO is weaker / needs to improve

- Measurement robustness: this paper’s “11 paraphrases per statement” is a concrete, easy-to-adopt protocol that many evaluations (often including ours) underuse.
- Multilingual breadth: 14 languages exposes failure modes and bias shifts that single-language evaluations will miss.
- Intervention story: activation-based steering suggests internal control options beyond prompting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “paraphrase robustness” component to GALILEO’s main evaluation (e.g., 5–10 paraphrases per item) and report variance, not just mean.
- [ ] If relevant, add a multilingual slice (even 3–5 languages) to test whether stability/stance phenomena are language-conditioned.
- [ ] In related work, cite as evidence that (a) stance-like properties scale with model size and (b) can be steered via simple activation interventions.

## Quotes / details to potentially cite

- Abstract (problem framing): LLMs show political biases, but prior work covers narrow models/languages; few examine controllability.
- Abstract (protocol numbers): “seven models … across 14 languages … Political Compass Test with 11 semantically equivalent paraphrases per statement”.
- Abstract (main findings): “larger models consistently shift toward libertarian-left … center-of-mass activation intervention … reliably steers … across multiple languages.”
