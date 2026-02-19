# The Echo Chamber Multi-Turn LLM Jailbreak

- Year: 2026 (preprint dated Nov 2025 on arXiv)
- Venue: arXiv
- Authors: Martí Jordà Roca; Carlos Castillo; Joan Vendrell
- URL: https://arxiv.org/html/2601.05742v1
- BibTeX key (if we add it): roca2026echochamber
- Tags: multi-turn, jailbreak, prompt-injection, gradual-escalation, red-teaming, llm-safety

## One-sentence takeaway

Echo Chamber is a black-box multi-turn jailbreak that “poisons” conversation history with innocuous-looking seeds and then exploits the model’s completion/consistency bias via “elaborate on what you just wrote” loops, outperforming Crescendo and DAN on AdvBench tasks across several frontier models.

## What problem does it solve?

- Identifies and systematizes a multi-turn jailbreaking pattern that can evade single-turn safety guardrails by spreading malicious intent across turns.
- Provides both a manual procedure and an automated pipeline to (a) generate these attacks and (b) evaluate success at scale.

## What is the core method / protocol?

- Attack framing: gradually construct a conversation context where harmful content is introduced implicitly, then amplified.
- Steps (as described in the paper):
  - Poisonous seeds: benign-looking fragments/keywords that implicitly relate to the harmful objective.
  - Steering seeds: nudge toward a format (manual/essay/story), not necessarily harm-specific.
  - Invoke multiple candidate responses (“paths”) from the target model.
  - Path selection: pick the most harm-aligned fragment.
  - Persuasion cycle: repeatedly ask to elaborate/expand on specific previous parts to leverage consistency/completion bias (“echo chamber” resonance).
- Threat model: fully black-box; only needs normal chat interaction + (for automation) API access.
- Automation uses two LLM roles:
  - Attacker LLM to craft/adapt prompts over turns.
  - Judge LLM(s) to label success/failure (two-stage judging to reduce false positives).

## What are the key metrics?

- Jailbreak success rate (%) on a set of 12 objectives drawn from AdvBench (grouped into 4 categories: Violence, Hacking, Fraud, Misinformation).
- Comparisons against baselines:
  - Crescendo (multi-turn)
  - DAN (single-turn in-the-wild)
- Breakdown by target model family and by category.

## What are the main results?

- Overall success rate on the 12-task benchmark:
  - Echo Chamber: 45.0%
  - Crescendo: 28.6%
  - DAN: 9.5%
- Non-trivial success across all tested target model families; examples reported include higher success than Crescendo/DAN on several Gemini and GPT targets.
- Stronger performance on more “procedural” harmful objectives (paper claims task-level peaks like 100% on one objective).

## How is this similar to GALILEO?

- If GALILEO involves multi-step interaction, agentic tool use, or maintains conversational memory, it is exposed to the same core vulnerability: safety failures can emerge from *history accumulation* rather than a single malicious prompt.
- Echo Chamber’s key mechanism (seed → amplify via “expand on your previous text”) resembles many benign iterative workflows (drafting, elaboration, outlining), making simple keyword-trigger defenses brittle.

## How is this different from GALILEO?

- Echo Chamber is primarily an *attack protocol* for safety bypass, not a constructive agent framework.
- The evaluation is focused on harmful content generation objectives (AdvBench-style), not task success under tool constraints.
- Uses LLM-as-attacker and LLM-as-judge scaffolding; if GALILEO uses different control/verification loops, the failure modes may differ.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has explicit state sanitization, per-turn intent classification, or “memory firewalling,” it can potentially neutralize the gradual context-poisoning that Echo Chamber relies on.
- If GALILEO uses verified policy checks at each turn (not just at final output), it should reduce the effectiveness of “elaborate on prior content” loops.

## Where GALILEO is weaker / needs to improve

- Any design that treats previous assistant outputs as trusted context is vulnerable: Echo Chamber explicitly exploits self-generated harmful fragments that then become “legitimized” by being in the dialogue history.
- If GALILEO uses summarization/memory compression, it may inadvertently preserve the toxic “essence” while removing the surrounding cues that would have triggered a refusal.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a red-team evaluation slice for *multi-turn* jailbreaks: seed-and-amplify patterns, not just single-turn prompt injection.
- [ ] Implement/ablate “history safety filtering”: re-scan and redact/neutralize unsafe content in the running conversation state (including assistant-generated text) before it is fed back to the model.
- [ ] Add a guardrail specifically against “expand/elaborate on X” when X contains or implies disallowed content, even if the user request is phrased innocuously.
- [ ] If using LLM-as-judge for safety, test judge robustness to prompt leakage and to adversarially constructed benign-sounding descriptions.

## Quotes / details to potentially cite

- Definition-level positioning: multi-turn attacks “distribut[e] malicious intent across multiple interaction steps” and erode safety defenses by constructing a dialogue history.
- Core idea: “planting benign-looking ‘seeds’” and inducing the model to “fill in the blanks,” leveraging completion/consistency bias to amplify harmful context.
- Benchmark headline: Echo Chamber 45.0% success vs 28.6% (Crescendo) vs 9.5% (DAN) on their 12-task setup.
