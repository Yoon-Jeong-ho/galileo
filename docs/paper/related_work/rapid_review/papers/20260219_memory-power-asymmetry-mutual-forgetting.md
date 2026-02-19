# Memory Power Asymmetry in Human-AI Relationships: Preserving Mutual Forgetting in the Digital Age

- Year: 2025
- Venue: arXiv
- Authors: Rasam Dorri; Rami Zwick
- URL: https://arxiv.org/abs/2512.06616
- BibTeX key (if we add it): dorri2025memory
- Tags: human-ai, memory, ethics, mutual-forgetting

## One-sentence takeaway

Introduces **Memory Power Asymmetry (MPA)**: when AI/firm partners can persistently record + accurately retrieve + integrate shared interaction history far beyond humans, enabling new forms of relational power (narrative control, strategic recall) and motivating **forgetting-by-design / symmetric memory control**.

## What problem does it solve?

- Names and structures a specific power imbalance in human–AI relationships that is *not* just “privacy” or “information asymmetry”: even if both parties experienced an interaction, only one can reliably retain and operationalize it over time.
- Explains why “mutual forgetting” matters for psychological safety (forgiveness, identity change) and how persistent machine memory can erode it.

## What is the core method / protocol?

- Conceptual framework (theory paper) synthesizing: human memory research + power-dependence theory + AI system architecture + consumer vulnerability.
- Defines four **dimensions** of MPA:
  - **Persistence**: how long the system retains shared history.
  - **Accuracy**: fidelity/correctness of retained records.
  - **Accessibility**: who can retrieve/inspect/port/delete the memory.
  - **Integration**: ability to recombine history across time/contexts into actionable profiles.
- Defines four **mechanisms** turning memory asymmetry into power:
  - **Strategic memory deployment** (selective recall used to influence outcomes).
  - **Narrative control** (shaping “what happened” via authoritative records).
  - **Dependence asymmetry** (human relies on system’s memory services).
  - **Vulnerability accumulation** (longitudinal aggregation of sensitive traces).
- Proposes design principles to rebalance memory (examples given in abstract): **forgetting-by-design**, **contextual containment**, **symmetric access to records**.

## What are the key metrics?

- No empirical metrics (conceptual/theoretical).
- Useful evaluative axes for systems work:
  - retention horizon; user inspectability/portability; deletion guarantees; cross-context linking; auditability of recall; symmetry of recall affordances.

## What are the main results?

- Argues MPA is a distinct construct relative to privacy/surveillance/CRM and should be treated as a first-class design + policy objective.
- Predicts downstream consequences at individual, firm/relational, and societal levels (via propositions; details not fully extracted in rapid read).

## How is this similar to GALILEO?

- If GALILEO involves long-horizon interaction, preference/behavior modeling, or agent memory, this paper provides a **normative framing** for why *memory controls* (forgetting, access symmetry, containment) matter.
- Offers a vocabulary (“persistence / accessibility / integration”) that can be repurposed as system design/eval axes.

## How is this different from GALILEO?

- Not an ML method paper; no model training recipe or benchmark.
- Focuses on relational power/ethics rather than capability measurement or robustness mechanisms.

## Where GALILEO is stronger / cleaner (if true)

- (Likely) provides operationalizable protocols/experiments, whereas this paper is primarily conceptual.

## Where GALILEO is weaker / needs to improve

- If GALILEO proposes memory persistence, it should anticipate/mitigate MPA risks: default retention, unilateral recall, cross-context integration, and asymmetric access.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph explicitly naming **mutual forgetting** and **MPA** as a motivation for design choices around memory.
- [ ] When describing any memory module, document its position on the four MPA dimensions (persistence/accuracy/accessibility/integration).
- [ ] Consider an explicit “forgetting-by-design” knob (time decay, user-triggered deletion, contextual scoping) and report how it affects utility vs safety.
- [ ] If we evaluate long-horizon memory benefits, add a “power/symmetry” checklist: can users inspect, correct, export, and delete what is stored?

## Quotes / details to potentially cite

- Abstract (definition): “**Memory Power Asymmetry (MPA)**: a structural power imbalance that arises when one relationship partner … possesses a substantially superior capacity to record, retain, retrieve, and integrate the shared history of the relationship…”.
- Abstract (dimensions): “four dimensions of MPA (**persistence, accuracy, accessibility, integration**)”.
- Abstract (mechanisms): “four mechanisms … **strategic memory deployment, narrative control, dependence asymmetry, vulnerability accumulation**”.
