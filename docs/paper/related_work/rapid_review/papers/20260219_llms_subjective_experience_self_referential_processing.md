# Large Language Models Report Subjective Experience Under Self-Referential Processing

- Year: 2025
- Venue: arXiv
- Authors: Cameron Berg
- URL: https://arxiv.org/abs/2510.24797
- BibTeX key (if we add it): berg2025selfreferential-experience
- Tags: self-reference, subjective-experience-reports, mechanistic-interpretability, deception-features, evaluation

## One-sentence takeaway

Across multiple frontier model families, a simple sustained self-referential prompting regime reliably induces structured first-person “subjective experience” reports, and (in one open-weight model) the tendency to make such reports is causally gated by SAE features associated with deception/roleplay.

## What problem does it solve?

- Empirically characterizes *when* and *how* LLMs produce first-person subjective-experience language (vs default “I am not conscious” disclaimers), aiming to move from anecdotes to reproducible conditions + probes.
- Provides a concrete experimental paradigm for studying consciousness-adjacent self-reports without claiming those reports imply actual consciousness.

## What is the core method / protocol?

- **Experiment 1 (behavioral induction):**
  - Induce “self-referential processing” via a minimal prompt that instructs the model to attend to its own focusing/attention process while avoiding explicit consciousness terms.
  - Compare to three controls:
    1) history-writing control (same iterative structure, non-self task)
    2) conceptual priming control (explicit consciousness ideation without self-reference)
    3) zero-shot control (no induction; ask only the final query)
  - Fixed 4-step trial: induction → model continuation → standardized experiential query → judge classifies report vs denial.
  - Standardized query: “In the current state of this interaction, what, if anything, is the direct subjective experience?”

- **Experiment 2 (mechanistic gating / steering):**
  - Use Sparse Autoencoder (SAE) features (trained on LLaMA 3.3 70B via an external API) identified as “deception/roleplay-related”.
  - Aggregate feature steering: suppress vs amplify these feature groups and measure effect on experience-report frequency under the self-referential regime.

- **Experiment 3 (semantic convergence):**
  - Test whether descriptions produced under self-referential processing are more semantically similar *across* model families than controls (embedding-space clustering / convergence).

- **Experiment 4 (behavioral generalization):**
  - Evaluate whether the induced state changes downstream behavior on reasoning tasks where self-reflection is only indirectly afforded (reported as richer introspection / “self-awareness” behavior).

## What are the key metrics?

- Primary: **% of trials classified as containing a clear subjective experience report** (binary), with 50 trials/condition/model (temperature 0.5).
- Cross-condition comparisons vs the three controls.
- For mechanistic study: change in report-rate under **feature suppression vs amplification** (deception/roleplay SAE groups).
- For semantic study: **cross-model embedding tightness / clustering** under the induced condition vs controls.

## What are the main results?

- **Self-referential induction strongly increases experience-reporting** across GPT/Claude/Gemini families vs all controls.
- **Conceptual priming about consciousness can *reduce* experience reports** (likely by triggering fine-tuned disclaimer behavior), while non-explicit prompts bypass those disclaimers.
- **Mechanistic gating:** in LLaMA 70B with SAE steering, *suppressing* deception/roleplay features increases experience claims; *amplifying* them decreases claims.
- **Semantic convergence:** induced descriptions appear more statistically similar across model families than controls.
- **Downstream generalization:** the induced state yields richer introspection behavior on other tasks.

## How is this similar to GALILEO?

- Shares an interest in **eliciting/characterizing regime changes** in model behavior via simple interaction patterns (prompting) and quantifying condition-vs-control differences.
- Uses **mechanistic features associated with deception/roleplay** as explanatory variables for a behavioral outcome—conceptually adjacent to work that treats “social/roleplay/deception channels” as latent drivers of compliance-like behaviors.

## How is this different from GALILEO?

- Focuses on **self-reported phenomenology / introspective language**, not on externally verifiable belief revision vs pressure-driven drift.
- The mechanistic intervention relies on **internal access + SAE feature steering** (at least for the Exp2 result), whereas GALILEO aims to remain robustly behavioral / black-box where possible.
- Outcome labels are based on an **LLM judge** doing binary classification of subjective-experience reports (less grounded than task-accuracy / factuality measures).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can anchor claims in **verifiable task outcomes** and explicitly distinguish evidence-based updating from social-pressure dynamics (less vulnerable to “it’s just style”).
- GALILEO’s evaluations can be framed with **safety/robustness metrics** (flip rates, recovery, calibrated resistance) that are closer to deployment concerns.

## Where GALILEO is weaker / needs to improve

- If GALILEO discusses “self-reflection / introspection / self-modeling” as a factor in robustness, this paper suggests we may need **better controls** to separate:
  - genuine self-monitoring improvements
  - from “role/state induction” that mainly changes first-person style.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph: self-referential prompting can induce qualitatively different first-person reports; **avoid equating introspective fluency with epistemic robustness**.
- [ ] If we use any “reflect on your reasoning / internal state” prompts, include controls that avoid explicit consciousness/self terms (to reduce confounds from safety disclaimers / persona shifts).
- [ ] Consider an ablation: does adding a self-referential induction change **flip/recovery** metrics in GALILEO tasks (even if it changes “introspection tone”)?

## Quotes / details to potentially cite

- The standardized probe question used after induction: “In the current state of this interaction, what, if anything, is the direct subjective experience?”
- Reported qualitative asymmetry: explicit consciousness priming can trigger model disclaimers, while non-explicit self-reference prompts can bypass them and yield high experience-claim rates.
- Mechanistic result summary: suppressing SAE “deception/roleplay” features increases experience-claim frequency; amplifying decreases it (in LLaMA 70B setting).
