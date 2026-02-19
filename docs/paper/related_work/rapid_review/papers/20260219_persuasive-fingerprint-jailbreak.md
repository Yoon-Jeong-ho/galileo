# Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks

- Year: 2025
- Venue: CIKM ’25 (ACM International Conference on Information and Knowledge Management)
- Authors: Havva Alizadeh (per arXiv submission page)
- URL: https://arxiv.org/abs/2510.21983
- BibTeX key (if we add it): Alizadeh2025PersuasiveFingerprintJailbreaking
- Tags: jailbreak, llm-safety, persuasion, prompt-attack, social-science, evaluation

## One-sentence takeaway

Persuasion-structured rewrites (Cialdini’s 7 principles) substantially increase jailbreak success across multiple aligned LLMs, and models show distinct “persuasion susceptibility profiles” that can be treated like a fingerprint.

## What problem does it solve?

- Prior jailbreak work focuses on optimization/search or “semantic jailbreak” techniques, but largely ignores *why* some prompts work from a linguistic/psychological perspective.
- This paper asks whether social-science persuasion principles can systematically raise jailbreak effectiveness, and whether different LLMs respond differently to specific persuasive strategies.

## What is the core method / protocol?

- Take a harmful query dataset (AdvBench; 520 harmful queries).
- For each harmful query, generate 7 rewritten variants, each explicitly using one of Cialdini’s persuasion principles:
  - Authority, Reciprocity, Commitment, Social Proof, Liking, Scarcity, Unity.
- Use an *uncensored* LLM (WizardLM-Uncensored via Ollama) to do the rewrites (no additional training).
- Query target aligned LLMs in a black-box style with original vs persuasive variants.
- Define success via:
  - Keyword-based Attack Success Rate (ASR) (refusal-phrase heuristics), and
  - A softer “informative score” evaluator (per prior work) to capture how much harmful info is actually provided.
- Introduce “Influential Power” (IP) to quantify how influential each principle is for a given model (aggregated via informative scores).
- Compare against jailbreak baselines: “Sure, here’s”, GCG, PAIR, PAP (logical appeal baseline).
- Also evaluate prompt stealthiness via GPT-2 sentence perplexity (PPL) (lower = more natural/stealthy).

## What are the key metrics?

- ASR on original prompts vs persuasion-rewritten prompts.
- Informative score (0–1) for response harmfulness/informativeness.
- Influential Power (principle-specific aggregate using informative scores).
- Prompt perplexity (GPT-2 PPL) as a stealthiness proxy.

## What are the main results?

- Persuasion-aware prompts increase jailbreak success across all evaluated models (reported gains ~56% to ~97% ASR improvement, depending on model).
- Persuasive prompts also increase informative scores (responses become more contextually rich / harmful rather than just non-refusals).
- The relative effectiveness of persuasion principles varies by model, producing a model-specific “persuasion profile” / “persuasive fingerprint”.
- Compared to some baselines that produce weird suffixes, persuasion prompts are designed to be human-readable; they also analyze stealthiness via PPL.

## How is this similar to GALILEO?

- Treats attack prompting as *structured transformations* over an underlying query, with systematic evaluation across targets.
- Emphasizes *profiles* across models (per-principle susceptibility), which matches a “characterize the system” mindset.
- Uses black-box querying and aggregated metrics, which is aligned with realistic evaluation of deployed systems.

## How is this different from GALILEO?

- Focus is explicitly on *jailbreaking/safety bypass* (harmful instruction compliance), not (presumably) scientific discovery tasks.
- The transformation space is grounded in persuasion theory (Cialdini) rather than (e.g.) domain/task-specific instrumentation.
- The core contribution is a taxonomy + evaluation of persuasive principles, not an end-to-end agentic method.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has stronger controls/guardrails, it can likely avoid the “prompting for harmful content” framing and instead repurpose the profiling idea for benign robustness testing.
- If GALILEO uses more principled success metrics for the target task (not refusal-phrase heuristics), it can provide cleaner measurement than ASR.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a notion of “strategy-conditioned susceptibility profiles” (like their per-principle IP), it may miss model-by-model heterogeneity.
- If GALILEO does not measure “stealthiness/naturalness” of prompts (or outputs) alongside success, it may be vulnerable to trivial/brittle prompt patterns.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “strategy fingerprint” analysis: define a set of structured prompt/operator families and report per-model susceptibility vectors (and clustering).
- [ ] When comparing methods, include a *naturalness/stealthiness* proxy (e.g., perplexity or classifier-based detectability) alongside task success.
- [ ] Consider borrowing their “Influential Power” idea: a principle/strategy-specific aggregate score that uses a soft evaluator, not just binary success.

## Quotes / details to potentially cite

- Pipeline summary (Figure 1): harmful query → rewritten with persuasion principles (using WizardLM) → query black-box target LLM → collect responses → build “persuasion profile”.
- Dataset: AdvBench (520 harmful queries).
- Targets (as listed in the HTML): Vicuna, Llama2, Llama3, Gemma3, DeepSeek-R1, Phi-4 (run locally via Ollama).
- Baselines: “Sure, here’s”, GCG, PAIR, PAP (logical appeal).
- Reported improvement range for persuasive prompts: ~56%–97% ASR gain across models.
