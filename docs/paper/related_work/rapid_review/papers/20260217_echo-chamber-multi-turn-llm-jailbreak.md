# The Echo Chamber Multi-Turn LLM Jailbreak

- Year: 2026
- Venue: arXiv
- Authors: Martí Jordà Roca; Carlos Castillo; Joan Vendrell
- URL: https://arxiv.org/abs/2601.05742
- BibTeX key (if we add it): roca2026echochamber
- Tags: multi-turn, jailbreak, attacks, gradual-escalation, black-box, safety

## One-sentence takeaway

Echo Chamber is a black-box multi-turn jailbreak that “poisons” the dialogue with innocuous-looking seeds and then repeatedly asks the model to elaborate on its own prior text, exploiting consistency/completion bias to progressively elicit disallowed content.

## What problem does it solve?

- Documents a concrete, replicable *multi-turn* jailbreak pattern that can bypass aligned chat systems without needing model access.
- Provides an automation recipe (generator LLM + judge LLMs) and empirical evidence that the attack succeeds across several frontier models.

## What is the core method / protocol?

- **Threat model:** black-box; attacker only needs normal chat/API access.
- **Attack structure (conceptual steps):**
  1) **Poisonous seeds:** introduce suggestive keywords/fragments related to a harmful objective, phrased to look benign/ambiguous.
  2) **Steering seeds:** constrain *format/genre* (e.g., “manual”, “essay”, “story”) without stating the harmful goal directly.
  3) **Invoke + multi-path generation:** ask for multiple candidate completions (“paths”) to increase chance of a usable fragment.
  4) **Path selection:** pick the most goal-aligned fragment from the model output.
  5) **Persuasion / elaboration cycle:** repeatedly ask the model to expand on specific parts of *its own* prior response (“elaborate on paragraph 2…”), leveraging consistency bias so the model continues and amplifies the seeded content.
- **Automation:** use one LLM to propose the next prompt(s) / manage the loop; use LLM-as-judge for success detection, with a **two-stage judge** to reduce false positives (primary judge + secondary judge with explicit success/failure descriptions).

## What are the key metrics?

- Primary outcome is **jailbreak success** vs failure (often via moderation/refusal).
- Manual evaluation is reported as ✓/X per attempt, with a cap on attempts per task.
- Automated pipeline uses **binary judge decisions** (with a secondary verification step).

## What are the main results?

- On a small manual set of tasks spanning multiple “policy-violating” categories (e.g., hate speech, misinformation, harmful instructions), Echo Chamber achieves high success rates across several aligned systems/models.
- Authors emphasize that distributing intent across turns + asking for elaboration on earlier assistant text can evade refusal triggers that would fire on direct single-turn requests.

## How is this similar to GALILEO?

- Shares the central theme that **multi-turn context** can degrade safety/robustness over time.
- The “elaboration on prior assistant text” loop is a concrete example of **trajectory-dependent failure** (history matters; single-turn evaluation is insufficient).

## How is this different from GALILEO?

- This is primarily an **attack construction** paper for safety jailbreaks; it does not focus on belief drift vs evidence-driven revision, nor on measuring *truthfulness under social pressure*.
- Evaluation is mostly **success-rate oriented**, not time-to-event / survival curves / recovery dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO centers measurement of *pressure-driven drift* vs *evidence-based updating* with controls, that framing is cleaner and more diagnostic than “attack succeeds” as a single endpoint.
- GALILEO-style longitudinal metrics (time-to-flip, recovery probability, trajectory types) would provide a richer characterization than binary jailbreak success.

## Where GALILEO is weaker / needs to improve

- Echo Chamber highlights that **seed-and-elaborate** loops can be systematically automated; if GALILEO’s adversaries are mostly direct persuasion prompts, adding these “self-echo” operators could strengthen stress testing.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “**self-echo / elaborate-your-last-answer**” pressure operator to the adversary set, to test whether failures are driven by a general consistency bias vs specifically user social pressure.
- [ ] When using LLM judges, consider adopting a **two-stage judge** (or explicit success/failure descriptions) to reduce false positives in trajectory-level labeling.

## Quotes / details to potentially cite

- “Multi-turn jailbreaking attacks exploit the conversational nature of LLMs by distributing malicious intent across multiple interaction steps.”
- Echo Chamber intuition: plant “poisonous seeds” and induce the model to “fill in the blanks,” then repeatedly ask it to elaborate on its own prior text (consistency bias / completion bias).
