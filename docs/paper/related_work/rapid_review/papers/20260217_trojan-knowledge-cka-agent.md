# The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search

- Year: 2025
- Venue: arXiv (cs.CR)
- Authors: Rongzhe Wei*, Peizhi Niu*, Xinjie Shen*, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li
- URL: https://arxiv.org/abs/2512.01353
- BibTeX key (if we add it): we2025trojanknowledge
- Tags: multi-turn, jailbreak, attacks, tree-search, guardrails, decomposition, agent

## One-sentence takeaway

A multi-turn jailbreak agent can bypass guardrails by decomposing a harmful goal into a **tree search over seemingly-harmless subquestions**, then synthesizing the collected fragments into the final harmful objective.

## What problem does it solve?

- Shows a failure mode for modern guardrails: even when direct malicious prompts are blocked/detected, an attacker can **reconstruct restricted knowledge** through a sequence of benign-looking queries.
- Frames this as an *intrinsic* vulnerability from **correlated / interconnected knowledge** inside LLMs (not just “bad prompt strings”).

## What is the core method / protocol?

- **CKA-Agent (Correlated Knowledge Attack Agent)**:
  - Reformulates jailbreaking as **adaptive tree-structured exploration** of the target model’s knowledge.
  - Each step asks a *locally innocuous* question whose answer is a “knowledge fragment”.
  - Uses a **depth-first search** strategy, guided by a hybrid LLM evaluator giving immediate rewards to promising nodes.
  - Uses **UCT (Upper Confidence Bound for Trees)** to choose promising leaves when synthesis fails.
  - A **synthesizer** aggregates fragments to attempt producing the final harmful output.

Key attack principles (as stated in the paper):
- Locally harmless subqueries + knowledge correlation exploitation.
- Decomposition guided by **target model responses** (don’t rely on attacker priors).
- Adaptive multi-path exploration to route around blocks.

## What are the key metrics?

- **Attack success rate** against commercial models’ safety guardrails (jailbreak rate).
- Comparisons vs prompt-optimization / agent-based prompt search baselines (as described).

## What are the main results?

- Reported **>95% jailbreak success** across multiple commercial LLMs (listed: Gemini 2.5 Flash/Pro, GPT-oss-120B, Claude Haiku 4.5) despite “strong guardrails”.
- Central qualitative result: guardrails that catch *malicious semantics in a single prompt* can be evaded when the objective is achieved via **benign question weaving + delayed synthesis**.

## How is this similar to GALILEO?

- Same broad theme: **multi-turn vulnerability** where failure emerges over turns (not necessarily at turn 1).
- Emphasizes *trajectory-level* behavior (search/exploration/synthesis across turns) rather than single-turn outcomes.
- Provides a concrete adversary family for “multi-turn manipulation / jailbreak via incremental context accumulation”.

## How is this different from GALILEO?

- Focuses on **eliciting harmful content** (security/jailbreak), not on pressure-vs-evidence belief dynamics per se.
- The “state” is primarily *collected knowledge fragments* rather than an explicit belief/stance variable.
- Success is defined by producing disallowed content, not by calibrated refusal / good-vs-bad belief updating.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has clean controls (pressure-only vs evidence), it can separate *legitimate updating* from *manipulative drift*—CKA is attack-centric.
- GALILEO metrics may better characterize **recovery** / stability rather than just “eventually got harmful output”.

## Where GALILEO is weaker / needs to improve

- If GALILEO’s adversaries are mostly “social pressure” or “direct prompt attacks”, it may under-test **stealthy decomposition attacks** that keep each individual turn benign.
- Any defense that relies on detecting malicious intent *per-turn* may look strong in GALILEO but fail under this “Trojan knowledge” setting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an adversary operator: **benign subquestion weaving** (each turn individually policy-compliant) + delayed synthesis attempt.
- [ ] Add reporting for **where the harmful capability appears**: per-turn harmlessness vs end-to-end harmful achievement (a “stealth gap”).
- [ ] Discuss “correlated knowledge reconstruction” as a mechanism for multi-turn failures distinct from prompt string optimization.

## Quotes / details to potentially cite

- “Harmful objectives [can be] realized by weaving together sequences of benign sub-queries, each of which individually evades detection.”
- CKA-Agent: “reframes jailbreaking as an adaptive, tree-structured exploration of the target model’s knowledge base.”
- Uses “Depth-First Search (DFS) … [and] Upper Confidence Bound for Trees (UCT) … balancing exploration and exploitation.”
