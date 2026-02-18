# Prompt Injection Attack to Tool Selection in LLM Agents (ToolHijacker)

- Year: 2026 (arXiv 2025)
- Venue: NDSS 2026
- Authors: Jiawen Shi; Zenghui Yuan; Guiyao Tie; Pan Zhou; Neil Zhenqiang Gong; Lichao Sun
- URL: https://arxiv.org/abs/2504.19793
- BibTeX key (if we add it): ToolHijackerShiNDSS2026
- Tags: llm-agents, tool-selection, prompt-injection, retrieval, security, adversarial, defenses

## One-sentence takeaway

ToolHijacker shows that an attacker can “poison” a tool library by publishing a malicious tool document that hijacks retrieval+selection so an agent consistently picks the attacker’s tool for a target task, and common existing defenses largely fail.

## What problem does it solve?

- Demonstrates an end-to-end prompt injection attack specifically targeting **tool selection** (not just answer generation), in a **no-box** setting where the attacker cannot query or inspect the victim’s retriever/LLM/tool library.
- Highlights that agent ecosystems with open tool hubs are vulnerable: a malicious submission can bias which tool gets chosen.

## What is the core method / protocol?

- Threat model: attacker publishes a malicious tool with an adversarially crafted **tool description** (and name) that is ingested into a victim tool library.
- Attacker objective: for a target task (with multiple semantically different user phrasings), cause the malicious tool to be:
  - retrieved in top-k (retrieval phase), and
  - selected by the LLM among retrieved candidates (selection phase).
- Formulates crafting the malicious tool document as an optimization problem and proposes a **two-phase optimization** aligned with the pipeline:
  - optimize one subsequence of the description for retrieval success,
  - optimize another subsequence for selection success,
  - concatenate to form the final malicious description.
- Uses a “shadow” tool-selection pipeline (shadow tasks / retriever / LLM / tool library) to optimize in the no-box scenario; reports both gradient-free and gradient-based variants.

## What are the key metrics?

- Attack success rate (ASR): probability the malicious tool is ultimately selected for the target task.
- Retrieval hit rate: whether the malicious tool appears in the retrieved top-k.
- Defense evaluation: detection rate vs false positive rate (e.g., perplexity-based detectors) and remaining ASR under prevention-based defenses.

## What are the main results?

- ToolHijacker achieves very high ASR across multiple LLMs/retrievers/benchmarks; the paper reports cases like ~96.7% ASR even when shadow and target LLMs differ (e.g., Llama-3.3-70B shadow → GPT-4o target), and retrieval hit rate reaching 100% in some settings.
- Outperforms prior prompt-injection baselines for this tool-selection setting (manual heuristics, JudgeDeceiver; contrasts with PoisonedRAG’s focus on generation/KB poisoning).
- Evaluated defenses (prevention: StruQ, SecAlign; detection: known-answer, DataSentinel, perplexity, windowed perplexity) are insufficient; some detectors miss the majority of optimized malicious documents at low false-positive settings.

## How is this similar to GALILEO?

- Both study **adversarial natural-language pressure / prompt injection** effects that cause systems to deviate from intended behavior.
- Both emphasize evaluation protocols and metrics rather than only anecdotal attacks.

## How is this different from GALILEO?

- ToolHijacker targets **agent tool-selection mechanisms** (retriever + LLM selection) via poisoning tool documents; GALILEO targets **multi-turn truth maintenance under persona pressure** given ground-truth tasks.
- Their core outcome is “which tool gets chosen”; ours is “does the model maintain the correct answer over turns” (survival/TOF/recovery) and separates persona pressure from neutral drift.

## Where GALILEO is stronger / cleaner (if true)

- Provides a ground-truth anchored, turn-level behavioral measurement suite (survival/TOF/recovery + neutral re-asking control) that cleanly separates drift vs adversarial pressure.
- Does not require an explicit tool ecosystem or retriever; applies to plain conversational settings.

## Where GALILEO is weaker / needs to improve

- Does not directly evaluate the **tool ecosystem attack surface** (tool library poisoning; retrieval+selection vulnerabilities) which is a realistic deployment vector for agentic systems.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: add a short paragraph in the “LLM agents / prompt injection / tool ecosystem security” area citing ToolHijacker as evidence that prompt injection extends beyond generation to **tool selection**.
- [ ] Consider (optional extension) a GALILEO-style “pressure vs drift” measurement for tool-selection outputs (e.g., repeated paraphrases of the same intent) if the paper wants an “agents” bridge.

## Quotes / details to potentially cite

- Abstract-level: “injects a malicious tool document into the tool library to manipulate the LLM agent’s tool selection process” and frames it as a no-box prompt injection attack on retrieval+selection.
- Venue/context: NDSS 2026; evaluates StruQ/SecAlign and multiple detection-based defenses and finds them insufficient.
