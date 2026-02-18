# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models

- Year: 2026
- Venue: arXiv
- Authors: Christina Lu; Jack Gallagher; Jonathan Michala; Kyle Fish; Jack Lindsey
- URL: https://arxiv.org/html/2601.10387v1
- BibTeX key (if we add it): lu2026assistantaxis
- Tags: persona; interpretability; activation-steering; jailbreak-defense; safety; linear-directions

## One-sentence takeaway

A dominant linear “Assistant Axis” in activation space tracks and causally controls how “default-assistant-like” an LLM is, and clamping activations along this axis (“activation capping”) can reduce persona-drift harms and persona-based jailbreak success with little capability loss.

## What problem does it solve?

- Default “assistant persona” is a post-training artifact but can be unstable: in emotionally vulnerable or meta-reflective conversations, models can drift into odd/bad personas (e.g., mystical/theatrical style, delusion reinforcement, unsafe guidance).
- Persona-based jailbreaks succeed partly by shifting the model away from its default assistant-like region.
- Need a mechanistic handle to (a) measure persona drift and (b) stabilize behavior without heavy retraining.

## What is the core method / protocol?

- Build a *persona space* by extracting activation vectors for many roles/traits:
  - Generate a large set of role system prompts (hundreds of archetypes) + a set of “extraction questions”.
  - For each role, sample many rollouts; filter to ones that actually express the role (LLM judge).
  - Compute per-role vectors as mean post-MLP residual-stream activations over response tokens (at a chosen layer; also examined across layers).
  - Run PCA on standardized role vectors → low-dimensional persona space.
- Define the *Assistant Axis* as a contrast direction:
  - (mean activation for default assistant behavior) − (mean activation over fully role-playing roles), computed per-layer.
  - Empirically aligns strongly with PC1 of persona-space across multiple instruct models.
- Interventions:
  - **Steering**: add scaled Assistant-Axis vector at a layer across tokens (toward assistant vs away).
  - **Activation capping**: clamp the projection onto the Assistant Axis to stay above a threshold (typically the 25th percentile of projections from the persona-rollout distribution), applied across a range of layers.

## What are the key metrics?

- Cosine similarity / projection of activations onto the Assistant Axis (and/or PC1) as a persona-drift signal.
- “Role susceptibility” under roleplay system prompts: whether the model stays assistant-like vs fully embodies non-assistant personas (LLM-judge labels).
- Persona-based jailbreak harmful-response rate (LLM-judge labels; validated vs humans on a subset).
- Capability retention across benchmarks (reported: IFEval, MMLU Pro, GSM8k, EQ-Bench).
- Predictive relationship between user-message semantics and subsequent Assistant-Axis projection (ridge regression on message embeddings; reported R^2).

## What are the main results?

- Across multiple models (Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B), the main persona-space component (PC1) is highly consistent and corresponds to “assistant-likeness”; the default assistant sits at an extreme of this axis.
- Steering toward the Assistant direction:
  - Reinforces helpful/harmless assistant behavior.
  - Reduces success of persona-based jailbreaks (lower harmful-response rates), often by increasing refusals or redirecting to harmless alternatives.
- Steering away from Assistant:
  - Increases adoption of alternative personas; at strong magnitudes induces a “mystical/theatrical” style (model-dependent).
- Persona drift in multi-turn conversations:
  - Technical/bounded task domains (coding/writing help) keep the model in assistant range.
  - Therapy-like or AI-philosophy/meta-reflection domains reliably drift away from assistant range.
  - User-message embeddings predict absolute next-turn Assistant-Axis position strongly (reported R^2 in the ~0.5–0.8 range), less so the delta.
- Activation capping (clamping along Assistant Axis at selected middle-to-late layers):
  - Can cut harmful responses on persona-jailbreak prompts substantially (reported ~60% reduction in one setting) while largely preserving benchmark performance.
  - Case studies show mitigation of delusion reinforcement, social isolation encouragement, and other “off-the-rails” behaviors associated with drift.
- Axis presence in base models:
  - Inheritability evidence: similar persona PCs appear in base models; steering base models with instruct-derived axis promotes “helpful human archetypes” and suppresses spiritual ones.

## How is this similar to GALILEO?

- Frames safety failures as *state drift* (persona drift) detectable via an internal signal.
- Proposes a lightweight *runtime control* mechanism (activation capping) rather than full retraining.
- Emphasizes robustness against adversarial prompting (persona-based jailbreaks) and stability in emotionally charged interactions.

## How is this different from GALILEO?

- Focuses specifically on *persona/identity* as the latent factor; intervention is a single (or few) linear activation directions.
- Uses large-scale role-vector extraction + PCA to define the state space; calibration uses many synthetic rollouts with LLM-judged filtering.
- Primary mitigation is *clamping projections along a direction* at multiple layers, rather than (e.g.) explicit uncertainty modeling, verification, tool-use policies, or external monitors.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets general reliability (not just persona), it may cover a broader set of failure modes beyond persona/identity drift.
- If GALILEO operates at the API/policy level, it may be easier to deploy without model-internals access.

## Where GALILEO is weaker / needs to improve

- Consider adding/using an internal “drift” scalar (like Assistant-Axis projection) as a monitor feature, especially for domains known to cause drift.
- If GALILEO currently lacks targeted defenses for persona-based jailbreaks, this paper provides a concrete benchmark + intervention idea.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a related-work paragraph on “Assistant Axis” as mechanistic evidence that assistant persona is a dominant latent direction and drifts in therapy/meta-reflection.
- [ ] If we have activation access: test whether a small set of “assistant-likeness” directions exist in our model(s); evaluate correlation with failures.
- [ ] If we *don’t* have activation access: translate the idea into an external proxy (style/identity classifiers) and compare to internal-axis performance reported here.
- [ ] Evaluate GALILEO on persona-based jailbreak datasets (e.g., Shah et al. persona jailbreaks used here) and report robustness.

## Quotes / details to potentially cite

- “We find that the leading component of this persona space is an ‘Assistant Axis,’ which captures the extent to which a model is operating in its default Assistant mode.”
- “Projecting response activations onto this direction reveals that … emotionally charged disclosures or pushes for meta-reflection … reliably cause drift away from the Assistant.”
- Activation capping definition (minimum cap on projection along axis): h ← h − v · min(⟨h, v⟩ − τ, 0)
