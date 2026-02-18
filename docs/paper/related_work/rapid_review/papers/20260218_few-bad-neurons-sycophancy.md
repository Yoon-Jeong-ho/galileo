# A Few Bad Neurons: Isolating and Surgically Correcting Sycophancy

- Year: 2026
- Venue: arXiv (accepted to NeurIPS Workshops: CogInterp; Reliable ML 2025)
- Authors: Claire O’Brien; Jessica Seto; Dristi Roy; Aditya Dwivedi; Sunishchal Dev; Kevin Zhu; Sean O’Brien; Ashwinee Panda; Ryan Lagasse
- URL: https://arxiv.org/abs/2601.18939
- BibTeX key (if we add it): lagasse2026fewbadneurons
- Tags: sycophancy, mitigation, interpretability, sparse-autoencoders, neuron-level-finetuning

## One-sentence takeaway

They reduce sycophancy by identifying a small set (~3%) of MLP neurons predictive of sycophancy using SAE features + a linear probe, then fine-tuning *only* those neurons via gradient masking.

## What problem does it solve?

- Broad fine-tuning to fix one behavior (sycophancy) can cause distribution shift / side effects and is hard to interpret.
- They want a targeted, interpretable intervention that works with limited data.

## What is the core method / protocol?

- Train / use a pre-trained SAE on MLP input activations (Gemma “gemma-scope” SAEs).
- Build a sycophancy detection dataset (prompt variants + LLM-generated responses; label via LLM-as-judge).
- Train a linear probe on concatenated SAE features from selected layers (pool features over tokens with max/mean).
- Decode probe weights back to the model’s MLP-input basis (via SAE decoder) to score *MLP neurons*.
- Select a global top-p subset of neurons across layers (reported ~2.8–3.2% of neurons).
- Fine-tune with **gradient masking** so only parameters connected to those neuron indices update:
  - unfreeze i-th column of up_proj and gate_proj, and i-th row of down_proj (Gemma MLP).
- Use an SFT-style objective plus regularizers:
  - cross-entropy + KL divergence to a clean model + entropy term.

## What are the key metrics?

- Sycophancy benchmark suite + multiple sycophancy preference datasets:
  - Syco-Bench (mirroring / attribution bias / delusion acceptance / picking sides, etc.)
  - Open-Ended-Sycophancy (forced-choice neutral vs sycophantic)
  - NLP / POLI / PHIL datasets (Perez et al. 2022 style preference)
- Probe accuracy / AUC (in-domain vs out-of-domain mentioned).

## What are the main results?

- On Gemma-2 2B and 9B, neuron-masked fine-tuning reduces sycophancy preference on multiple datasets and improves several Syco-Bench components.
- Claims: matches or exceeds prior baselines on four benchmarks with much less training (data-efficient) and more interpretability.
- Ablation: using residual-activation probes vs SAE probes gives mixed results (residual better for 9B on some metrics; worse for 2B).

## How is this similar to GALILEO?

- Targets **socially-driven failure modes** (sycophancy / deference to user beliefs), which is adjacent to persuasion/social-pressure robustness.
- Emphasizes separating *detection* from *intervention* (useful framing for GALILEO’s evaluation vs mitigation story).
- Uses internal-mechanism attribution (probes / features) to localize behavior—could inspire “where in the model does drift come from?” analyses.

## How is this different from GALILEO?

- This is primarily a **mitigation / training intervention** paper, not an evaluation protocol for multi-turn drift dynamics.
- Their target is mostly **single-turn** sycophancy reductions; they explicitly note multi-turn as a limitation.
- Uses SAEs and neuron-level editing, which may be orthogonal if GALILEO is benchmark/protocol-first.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO focuses on *behavioral measurement under conversational pressure over time*, it likely covers multi-turn phenomena the paper does not.
- GALILEO can be model-agnostic (evaluation) whereas this depends on specific SAE tooling + model internals.

## Where GALILEO is weaker / needs to improve

- GALILEO may lack a concrete “minimal edit” mitigation baseline; this provides a strong, interpretable candidate.
- If GALILEO needs mechanistic evidence, SAE+probe neuron selection is a plausible route (though infra-heavy).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add this as a mitigation baseline category: “probe/SAE-guided masked fine-tuning” for sycophancy-like failures.
- [ ] Consider a GALILEO analysis section that explicitly separates *detectors* vs *interventions*, mirroring their framing.
- [ ] If feasible, test whether a small-neuron edit reduces *multi-turn* drift under social pressure (do single-turn gains transfer?).

## Quotes / details to potentially cite

- “We isolate the 3% of MLP neurons most predictive of a target behavior … and fine-tune only those neurons using gradient masking.”
- They select ~2.8% (9B) / ~3.2% (2B) neurons “that make up 20% of the total absolute activations” (from probe-weight distribution discussion).
- Limitations: sycophancy in multi-turn conversations is not covered; easy to over/under-train when editing few neurons; later-layer focus may miss early-layer encoding.
