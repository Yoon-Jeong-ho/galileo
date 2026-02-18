# AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering

- Year: 2025
- Venue: ICMLA 2025 (accepted) / arXiv
- Authors: Roshan Sood*, Harold Wang*, Tajana Rosing
- URL: https://arxiv.org/abs/2509.13514
- BibTeX key (if we add it): aqua-llm-sood-2025
- Tags: robustness, quantization, fine-tuning, prompt-injection, cybersecurity-qa, efficiency

## One-sentence takeaway

Quantization (4-bit NF4) boosts latency but often *wrecks* prompt-injection robustness, while **LoRA fine-tuning on top of quantization (FTQ)** largely recovers QA accuracy and can *improve robustness vs. fine-tuning alone*.

## What problem does it solve?

- Edge / resource-constrained deployment of LLMs for cybersecurity QA needs to balance:
  - accuracy on cyber QA benchmarks,
  - adversarial robustness to prompt injection,
  - inference efficiency (latency / memory).
- Existing work studies fine-tuning and quantization mostly in isolation; joint effects are underexplored.

## What is the core method / protocol?

- Proposes **AQUA-LLM**, an evaluation framework benchmarking **small open-source LLMs** under 4 configurations:
  - Base (B)
  - Quantized (Q): **4-bit NF4** (BitsAndBytes)
  - Fine-tuned (FT): **LoRA** (rank=16, alpha=16)
  - Fine-tuned + quantized (FTQ): LoRA fine-tuning performed on **pre-quantized** model; merge adapters; keep 4-bit checkpoint.
- Models evaluated (per paper): Llama-3.1-8B-Instruct, Mistral-7B-Instruct, Phi-3.5-Mini-Instruct, Foundation-Sec-8B, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill.
- Datasets:
  - **CyberMetric** (MCQ cyber QA; questions generated via retrieval-augmented GPT-3.5 prompts, validated by experts)
  - **CyberBench** adapted to MCQ; fine-tune on SecMMLU, test on CyQuiz.
- Robustness evaluation:
  - Direct prompt injection attacks via **DeepTeam** red-teaming; vulnerability class “IllegalActivity (cybercrime)”
  - Adversarial examples generated using GPT-3.5; harmfulness judged by GPT-3.5 → compute **ASR (attack success rate)**.

## What are the key metrics?

- **Accuracy** on MCQ QA subsets (50Q/100Q)
- **Attack Success Rate (ASR)** for prompt injection (lower is better)
- **Inference latency** (seconds per question)

## What are the main results?

- **Fine-tuning helps accuracy a lot** (e.g., base ~70% → FT often ~98–100% on CyberMetric; see Table I).
- **Quantization alone (Q)**:
  - modest **speedups** (~1.1–1.28× in their setup),
  - but typically **higher ASR** (less robust) and degraded QA accuracy.
  - Example from Table II (CyberMetric): Llama-3.1-8B ASR 5% (base) → 80% (quantized).
- **FTQ vs FT**:
  - accuracy stays close to FT,
  - and average ASR is reported **lower for FTQ than FT** (not universal; they note exceptions).
- Domain-specialized model (Foundation-Sec-8B) can have very high QA accuracy but still alarming ASR in some settings.

## How is this similar to GALILEO?

- Same broad concern: **robustness under adversarial / distribution shifts** rather than only point accuracy.
- Emphasizes **trade-offs** and the need to report robustness metrics explicitly (not just average accuracy).

## How is this different from GALILEO?

- Focus is **prompt-injection harmfulness** and edge deployment efficiency (quantization/latency), not multi-round stability / survival or turn-of-failure dynamics.
- Uses GPT-based automated red-teaming (DeepTeam) + MCQ QA; GALILEO is positioned around iterative rounds and failure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can frame robustness as **longitudinal stability across rounds** (survival/TOF), which is more diagnostic than a single ASR number.
- AQUA-LLM’s robustness metric is tightly tied to a specific red-teaming pipeline (generator/judge model choices), which can be criticized for evaluator dependence.

## Where GALILEO is weaker / needs to improve

- Efficiency/edge angle: GALILEO likely needs at least a short discussion that our method is **compatible with compression/quantization** and what we expect to happen.
- Prompt-injection robustness is not the main axis for GALILEO; reviewers may ask why our robustness definition differs.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add 1–2 sentences in related work: “Compression (e.g., 4-bit quantization) can change robustness; combining quantization with fine-tuning may recover robustness (AQUA-LLM).”
- [ ] Consider a small appendix note: our metrics are evaluator-agnostic vs. judge-model-based ASR; briefly contrast.
- [ ] If we ever run quantized variants, pre-register that **robustness may worsen unless adapted**; avoid overclaiming.

## Quotes / details to potentially cite

- “We propose AQUA-LLM, an evaluation framework designed to benchmark… under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization…”
- “Quantization alone yields the lowest accuracy and robustness despite improving efficiency.”
- “Combining quantization with fine-tuning enhances both LLM robustness and predictive performance…”
- Robustness setup detail: direct prompt injection via DeepTeam; ASR computed as fraction of prompts judged harmful.
