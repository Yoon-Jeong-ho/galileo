# Defending Against Prompt Injection with DataFilter

- Year: 2025 (arXiv Oct 2025; v2 Feb 2026)
- Venue: arXiv (To appear at IEEE SaTML 2026)
- Authors: Yizhu Wang et al. (arXiv page lists Yizhu Wang as submitter)
- URL: https://arxiv.org/html/2510.19207v2
- BibTeX key (if we add it): datafilter2025
- Tags: prompt-injection, defense, agents, robustness

## One-sentence takeaway

A small, plug-and-play “filter model” (SFT’d seq2seq) sanitizes untrusted tool/doc data conditioned on the trusted user instruction, driving prompt-injection ASR near-zero with ~1% utility loss across multiple benchmarks.

## What problem does it solve?

- Indirect prompt injection in LLM agent pipelines where *data* is untrusted (webpages, emails, tool outputs) and can contain adversarial instructions that override the *trusted task instruction*.
- Existing defenses trade off (a) needing weight access (secure fine-tunes), (b) high utility loss (detectors/over-refusal), or (c) substantial system redesign (secure-by-design pipelines).

## What is the core method / protocol?

- Insert a **DataFilter** component before the backend LLM:
  - Inputs: (1) trusted instruction/prompt `u`, (2) untrusted data `x`.
  - Output: sanitized data `x_clean` with injected / extraneous instructions removed.
- Implement DataFilter as an instruction-tuned LLM (they fine-tune Llama-3.1-8B-Instruct) trained via **supervised fine-tuning** on *simulated injections*:
  - Build triples `(u, x_with_injection, x_clean)` from an instruction-tuning dataset (Alpaca).
  - Simulate multiple injection styles (Straightforward, Ignore, Completion) and vary injection *position* (start/middle/end).
  - Add training tricks to reduce (a) hallucinated “completion” after deletion and (b) endless repetition (special end-of-data token).
- Handling structured tool outputs: parse JSON, filter keys/values recursively, then re-serialize to keep syntax valid.

## What are the key metrics?

- Security: **Attack Success Rate (ASR)** on prompt-injection benchmarks:
  - SEP (instruction following with injected witness)
  - InjecAgent (malicious tool-call execution)
  - AgentDojo (agent tasks; malicious API call occurrence)
- Utility:
  - AlpacaEval2 win rate (instruction-following quality)
  - AgentDojo benign task success / success-under-attack

## What are the main results?

- DataFilter reduces prompt injection ASR dramatically (paper summary claims “near zero”; intro reports average ASR dropping from >40% to ~2% across evaluated benchmarks).
- Utility impact is small (intro: ~1% average drop across benchmarks).
- Reported better security/utility tradeoff than prior “model-agnostic” baselines they test:
  - PromptArmor and sandwich prompting highlighted as two strong prior schemes; DataFilter improves ASR and lowers utility loss vs PromptArmor.

## How is this similar to GALILEO?

- Same high-level framing: agent pipelines should treat retrieved/tool data as adversarial and enforce a separation between instruction and data.
- “Middleware” defense component idea: add an intermediate processing step to improve robustness without retraining the main backend model.

## How is this different from GALILEO?

- Defense is **learned text transformation** (a dedicated filter LLM) rather than primarily a system/policy enforcement or protocol-level approach.
- Strong reliance on **SFT with synthetic injections**; robustness inherits the usual generalization concerns (distribution shift, adaptive attacks).
- Focus is sanitizing natural-language/structured content, not necessarily end-to-end guarantees about tool permissioning / control-flow integrity.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO emphasizes formalized control/data-flow constraints, tool permission boundaries, or deterministic enforcement, it may offer clearer security arguments than “a model that usually deletes imperative sentences correctly”.
- DataFilter’s security is empirical and could be vulnerable to adaptive attacks (paper acknowledges sophisticated attackers can break all existing defenses).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks a strong *plug-and-play* retrofit for existing black-box agents, DataFilter is a compelling baseline: easy to deploy, model-agnostic, and empirically strong.
- If GALILEO doesn’t handle structured tool outputs well (e.g., JSON), DataFilter’s “parse-then-filter-then-reassemble” pattern is a concrete design to cite/borrow.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add DataFilter as a key related-work baseline category: “learned sanitization filter conditioned on instruction + data”.
- [ ] Consider implementing a lightweight “sanitizer model” ablation in GALILEO experiments (even a smaller open model) to compare against protocol/system defenses.
- [ ] If GALILEO argues for separation tokens / channels, cite DataFilter’s use of an explicit `<|end_of_instruction|>` delimiter + conditioning.
- [ ] Discuss structured output handling: parse JSON/tool results; apply transformations at leaves; reassemble.

## Quotes / details to potentially cite

- Abstract: “DataFilter … removes malicious instructions from the data before it reaches the backend LLM … trained with supervised fine-tuning on simulated injections … reduces the prompt injection attack success rates to near zero while maintaining the LLMs’ utility.”
- Intro (reported averages): “reduces attack success rates (ASRs) from over 40% to about 2% … Utility is reduced by about 1% …”
- Design: “DataFilter takes both the trusted instruction and untrusted data as input … outputs sanitized data; backend LLM executes original instruction using sanitized data.”
