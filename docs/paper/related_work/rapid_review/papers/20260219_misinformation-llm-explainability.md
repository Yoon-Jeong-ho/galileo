# Misinformation Detection using Large Language Models with Explainability

- Year: 2025
- Venue: ACAI 2025 (Proc. 8th International Conference on Algorithms, Computing and Artificial Intelligence)
- Authors: Thanh Thi Nguyen et al.
- URL: https://arxiv.org/abs/2510.18918
- BibTeX key (if we add it): nguyen2025misinformation
- Tags: misinformation-detection, transformers, roberta, distilbert, efficiency, interpretability, lime, shap

## One-sentence takeaway

A simple but well-engineered fine-tuning recipe (freeze-then-progressive-unfreeze + LLRD) plus LIME/SHAP explanations yields competitive misinformation detection, highlighting that DistilBERT can match RoBERTa at much lower compute.

## What problem does it solve?

- Scalable misinformation detection on text benchmarks, with an emphasis on (a) compute efficiency for deployment and (b) interpretability to improve user trust and auditability.

## What is the core method / protocol?

- Text preprocessing (remove hyperlinks/special chars/emojis/HTML; lowercase normalization).
- Train PLM classifiers (they mention testing multiple models; key comparison is RoBERTa vs DistilBERT).
- Two-phase training curriculum:
  - Phase 1: freeze transformer backbone; train only classification head.
  - Phase 2: progressively unfreeze layers while applying layer-wise learning rate decay (LLRD).
- Add interpretability:
  - LIME for local, token-level rationales.
  - SHAP for global feature attribution.
- Evaluate on two datasets with a “unified protocol” including stratified splits.

## What are the key metrics?

- Classification: Accuracy, Precision, Recall, F1, AUROC.
- Efficiency: parameter count, training time per epoch, inference latency, throughput.

## What are the main results?

- DistilBERT achieves accuracy comparable to RoBERTa (at least on COVID Fake News) while being substantially more efficient.
- They report concrete efficiency numbers (from their logs) for DistilBERT (see quotes).
- Interpretability is provided via LIME (token-level) and SHAP (global) “without compromising performance” (claim).

## How is this similar to GALILEO?

- If GALILEO targets trustworthy/transparent NLP systems: this is aligned in combining strong text encoders with explicit interpretability artifacts (local + global rationales).
- If GALILEO cares about practical deployment constraints: this paper frames an explicit efficiency-vs-accuracy trade-off and argues for lighter models.

## How is this different from GALILEO?

- Focuses on supervised misinformation classification on two specific datasets; not a general reasoning/verification framework.
- Uses post-hoc explainers (LIME/SHAP) rather than inherently interpretable modeling or retrieval/verification traces.
- Limited methodological novelty beyond a standard “freeze then fine-tune” + LLRD recipe and explanation tooling.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides faithful, model-grounded rationales (vs perturbation-based LIME) or verifiable evidence (retrieval/citations), that is likely stronger than post-hoc attributions.
- If GALILEO evaluates cross-domain robustness or distribution shift, that would exceed this paper’s relatively narrow benchmark scope.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently uses a heavy backbone only: this paper is a reminder to benchmark distilled/lightweight backbones and report throughput/latency explicitly.
- If GALILEO lacks standardized interpretability outputs: LIME/SHAP-style local+global explanations (or analogous artifacts) may be expected by reviewers for “trustworthy” claims.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “efficiency” table (params, training time/epoch, throughput, latency) for at least one lightweight backbone (e.g., DistilBERT-class) vs a larger backbone.
- [ ] Consider a freeze-then-unfreeze schedule with LLRD as a simple, defensible fine-tuning baseline.
- [ ] If we discuss explainability: explicitly separate local vs global explanations, and discuss faithfulness/limitations of post-hoc methods.

## Quotes / details to potentially cite

- “We adopt a training curriculum that first freezes the backbone to stabilize task adaptation, then progressively unfreezes layers with layer-wise learning rate decay, mitigating catastrophic forgetting and improving convergence.” (Section II)
- DistilBERT efficiency numbers reported: “training time is ∼397 s/epoch … inference throughput ∼71.8 samples/s and latency ∼13.9 ms/sample on 2,041 test items.” (Section II)
