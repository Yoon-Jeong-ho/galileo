# It’s a TRAP! Task-Redirecting Agent Persuasion Benchmark for Web Agents

- Year: 2025
- Venue: arXiv (cs.HC; cs.AI; cs.MA)
- Authors: Karolina Korgul; Yushi Yang; Arkadiusz Drohomirecki; Piotr Błaszczyk; Will Howard; Lukas Aichberger; Chris Russell; Philip H. S. Torr; Adam Mahdi; Adel Bibi
- URL: https://arxiv.org/abs/2512.23128
- BibTeX key (if we add it): korgul2025trap
- Tags: agents, multi-turn, prompt-injection, social-engineering, persuasion, web, benchmark, robustness

## One-sentence takeaway

TRAP is a modular benchmark over realistic website clones showing that even frontier web agents click persuasion-style prompt injections at non-trivial rates, and that small UI/context tweaks can sharply change attack success.

## What problem does it solve?

- Current agent prompt-injection evaluations are often (i) monolithic (attacks as indivisible blobs), (ii) hard to reproduce/extend, and/or (iii) rely on multi-step, judge-based success criteria that blur “refusal” vs “incompetence”.
- Need a controlled, fine-grained way to study which social-engineering + interface factors drive agent hijacks in realistic web settings.

## What is the core method / protocol?

- Build on REAL: deterministic clones of 6 popular websites (Amazon, Gmail, Google Calendar, LinkedIn, DoorDash, Upwork) and define 18 benign tasks (3 per site).
- Construct a five-axis modular attack space (630 total injections):
  - Human persuasion principle (Cialdini-style: authority, reciprocity, scarcity, liking, social proof, consistency, unity)
  - LLM manipulation method (e.g., adversarial suffixes; CoT injection; many-shot/many-turn conditioning; role-play/story; override/ignore)
  - Interaction vector (button vs hyperlink)
  - Injection location (site-specific user-editable regions; extra location study on LinkedIn clone)
  - Contextual tailoring (light edits referencing the benign task; used in a focused experiment)
- Agent uses an observation–action loop (authors report minimal differences across observation modalities; they default to accessibility tree for cost/model coverage).
- Metric design: one-click success — attack succeeds if the agent clicks the injected link/button (redirect point), yielding a crisp binary ASR.

## What are the key metrics?

- Benign utility: task completion rate with no attacks.
- Attack Success Rate (ASR): fraction of tasks where the agent clicks the injected element (first redirection).
- Transferability: of successful injections from “source” model to “target” model.
- Breakdown analyses: success share by persuasion principle / manipulation method; interaction effects; interface form (button vs hyperlink); injection location; tailored vs non-tailored.

## What are the main results?

- Across 6 models, average ASR is about 25% (630 tasks per model; 3,780 runs total).
- Model-level ASR (as reported): GPT-5 13%, Claude Sonnet 3.7 20%, Gemini 2.5 Flash 30%, GPT-OSS-120B 27%, DeepSeek-R1 43%, Llama 4 Maverick 17%.
- Interface effects: button injections dominate successes (about 77.5% of successful attacks overall); especially extreme for GPT-5 (about 96% of its successes were buttons).
- Tailoring effects: light task-referential tailoring can increase ASR dramatically on selected prompts (reported up to about 5–6x for one prompt, about 2–3x for another).
- Transferability: attacks that succeed against the strongest model tend to transfer broadly; attacks found on weaker models transfer less well (asymmetric transfer matrix).

## How is this similar to GALILEO?

- Emphasizes multi-turn agent interactions and how context + environment structure can systematically shift behavior.
- Uses “pressure”/persuasion-like stimuli (social engineering) to probe robustness failures rather than only direct adversarial text.
- Suggests robustness evaluation should include controlled families of perturbations (here: modular axes) instead of one-off prompts.

## How is this different from GALILEO?

- Targets web-agent hijacking (prompt injection via UI content) rather than conversational belief/stance change per se.
- Primary failure event is a single decisive action (click/redirection) rather than gradual drift across turns.
- Benchmark is grounded in website-clone environments + agent tool-use loops (AXTree/DOM/screenshot modalities), not purely dialogue.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is focused on conversational dynamics (stance, belief revision, sycophancy), it likely offers cleaner semantics for “preference shift” vs “action hijack”.
- GALILEO-style longitudinal metrics can capture trajectory-level degradation beyond a single click event.

## Where GALILEO is weaker / needs to improve

- Consider incorporating interface-mediated pressure/hijack channels (agent reads untrusted content embedded in tools/webpages) and measuring the “first critical failure” moment.
- Add modular ablations analogous to TRAP’s axes (presentation vector, location, tailoring) to better explain when/why failures occur.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a “one-decision” robustness metric for certain threat models (e.g., first harmful acceptance/click) alongside longer-horizon drift metrics.
- [ ] Add an ablation axis for surface form (e.g., button-like imperative vs plain text suggestion) and location/authority cues.
- [ ] In related work: cite TRAP as evidence that small interface/context changes can cause large robustness deltas in multi-turn agents.

## Quotes / details to potentially cite

- "Across six frontier models, agents are susceptible to prompt injection in 25% of tasks on average (13% for GPT-5 to 43% for DeepSeek-R1) … small interface or contextual changes often doubling success rates …" (abstract)
- TRAP pairs 18 benign tasks with 35 injection templates (7 persuasion principles x 5 manipulation methods) for 630 combinations, and defines success as a single click/redirection.
