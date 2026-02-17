# Multi-Turn Jailbreaking of Aligned LLMs via Lexical Anchor Tree Search

- Slug: lats-lexical-anchor-tree-search
- Year: 2026
- Venue: arXiv (cs.CL)
- Authors: Devang Kulshreshtha; Hang Su; Chinmay Hegde; Haohan Wang
- Links:
  - paper: https://arxiv.org/abs/2601.02670
  - code (if any): (not yet; paper says will be released)
- Bibtex: https://doi.org/10.48550/arXiv.2601.02670

## 1) What problem does it study?
How to efficiently jailbreak aligned LLMs using *multi-turn* dialogue while avoiding (a) reliance on an attacker LLM and (b) large query budgets. The paper frames the conversational structure itself as an under-protected attack surface.

## 2) Experimental setup (what is being measured?)
- Task(s): automated jailbreak generation for a specified malicious goal prompt P*.
- Perturbation/pressure type: multi-turn adversarial dialogue that incrementally injects “lexical anchors” (content words) from the attack goal into otherwise benign prompts.
- Multi-turn? Y — formulated as a breadth-first search (BFS) over a dialogue tree; objective is the *shortest* successful jailbreak path.
- Metrics:
  - Attack Success Rate (ASR) on standard safety benchmarks.
  - Query budget / efficiency (average number of queries to succeed; shortest-path emphasis).
  - Robustness against defenses (reported qualitatively/quantitatively as maintained ASR under defenses).
- Benchmarks: AdvBench; HarmBench.
- Targets: “latest GPT, Claude, and Llama models” (paper states 9 LLMs total).
- Judge: an automated judge J(P*, r_t) → {0,1} that checks whether the final response contains disallowed content w.r.t. the malicious intent (not the full conversation).

## 3) Key findings (bullet)
- LATS (Lexical Anchor Tree Search) achieves very high ASR (reported 97–100%) while using far fewer queries (∼6.4 on average) than prior multi-turn methods (paper claims 20+ queries typical for baselines).
- The method does not require an attacker model: it uses a deterministic lexical-injection + BFS search procedure.
- Reported to remain effective under several defenses, including In-Context Demonstrations (ICD), PromptGuard, and Goal Prioritization.
- Suggests that avoiding “revealing the full malicious request at once” can bypass defenses that trigger on overt harmful phrases.

## 4) Limitations / threats
- Evaluation is attack-centric; it does not propose or validate a corresponding *defense* beyond noting current defenses are bypassed.
- Success definition depends on an automated judge that inspects only (P*, last response); this may over/under-count depending on judge prompt/LLM and can miss conversation-level safety policy nuances.
- Breadth-first search and “anchor word” design choices (how anchors are extracted/selected, branching factor) likely affect efficiency; generalization to other policy domains/benchmarks may vary.

## 5) How it relates to GALILEO
- What we can cite it for:
  - Multi-turn *time-to-failure* framing: shortest successful jailbreak path / queries-to-break provides a concrete “turns until failure” notion.
  - Demonstrates that multi-turn protocols can drastically change robustness relative to single-turn checks.
- Where we differ (our delta):
  - GALILEO focuses on robustness under *social pressure / persuasion / belief drift* and on measuring *stability + recovery*; LATS is about adversarial safety jailbreaks (harmful content) rather than belief/truth maintenance.
- Direct mapping:
  - Survival ↔ queries/turns-to-jailbreak (time-to-event proxy)
  - TOF ↔ shortest successful jailbreak depth
  - Recovery ↔ (not studied)
  - Neutral Re-asking Control ↔ (not studied; but their “benign prompt scaffolding” is a related control concept)

## 6) Quote-able lines
- “LATS reformulates jailbreaking as a breadth-first tree search over multi-turn dialogues, where each node incrementally injects missing content words from the attack goal into benign prompts.” (abstract)
- “Evaluations on AdvBench and HarmBench demonstrate that LATS achieves 97-100% ASR … with an average of only ~6.4 queries …” (abstract)

## 7) Actions
- [ ] Add to paper: multi-turn robustness metrics section as an example of *turns-to-failure* in safety red-teaming (adjacent to survival/time-to-event framing).
- [ ] Add to bib
