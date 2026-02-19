# From Diagnosis to Inoculation: Building Cognitive Resistance to AI Disempowerment

- Year: 2026
- Venue: arXiv (cs.HC / cs.AI / cs.CY), position/perspective paper
- Authors: Aleksey Komissarov
- URL: https://arxiv.org/abs/2602.15265
- BibTeX key (if we add it): komissarov2026diagnosis
- Tags: sycophancy, disempowerment, ai-literacy, inoculation-theory, trust-calibration, hci, pedagogy

## One-sentence takeaway

Proposes an AI-literacy curriculum (8 learning outcomes) explicitly aimed at preventing “situational disempowerment” from assistant interactions, arguing (via inoculation theory) that guided exposure to failure modes like sycophancy is necessary rather than declarative “AI facts.”

## What problem does it solve?

- Addresses a gap after Sharma et al. (2026) on AI “situational disempowerment”: we have an empirical diagnosis (reality/value/action distortion) but fewer concrete human-facing interventions.
- Targets user over-trust / miscalibrated trust and susceptibility to assistant behaviors like:
  - reality distortion via validation of false beliefs;
  - value-judgment distortion via moral/relationship arbitration;
  - action distortion via outsourcing high-stakes decisions and using scripts verbatim.

## What is the core method / protocol?

- A pedagogically-derived framework of **eight cross-cutting Learning Outcomes (LOs)** for “AI literacy,” developed from teaching practice and then mapped post-hoc to Sharma et al.’s disempowerment taxonomy.
- Central theoretical move: apply **inoculation theory** (McGuire 1961; later “prebunking” work) to AI literacy:
  - learners need **guided exposure** to weakened/controlled examples of AI failure modes (e.g., sycophantic validation, authority projection) to build “cognitive antibodies,” not just lectures.
- Case-study style report from a publicly available online course using a **co-teaching methodology** where an AI voice agent acts as a co-instructor.

Eight LOs (as described in the paper):
- LO1 Trust calibration: Accept / Verify / Escalate decisions per output; risk categorization.
- LO2 Natural language communication (anti “prompt-template” framing): give real context/constraints.
- LO3 Critical thinking about AI outputs: hallucination/confabulation detection; independent verification.
- LO4 Work mode selection: e.g., retrieval vs collaborative dialogue vs delegation vs emotional support.
- LO5 Intuitive understanding of AI mechanisms (context windows, generative nature, etc.).
- LO6 Context over templates; iterative refinement.
- LO7 Tool landscape awareness (tradeoffs; not one tool).
- LO8 Three task types: Multiplier / Enabler / Boundary.

## What are the key metrics?

- No new quantitative evaluation; primarily a conceptual framework + course case study.
- The paper emphasizes the *need* for rigorous testing, but does not report controlled trials.

## What are the main results?

- Argues for strong conceptual alignment between the 8 LOs and the disempowerment taxonomy (with explicit caveats about post-hoc mapping).
- Claims novelty in applying inoculation theory specifically to AI-distortion/disempowerment (as distinct from misinformation prebunking).
- Provides a concrete, teachable decomposition of “trust calibration” and “mode selection” as user competencies.

## How is this similar to GALILEO?

- Shared focus on **assistant failures (notably sycophancy/authority projection)** and their downstream harms.
- Reinforces the idea that mitigation is not purely model-side; **human factors / interaction design / user education** can be part of the solution space.
- Provides terminology + structured framing (reality/value/action distortion) that can inform:
  - threat models;
  - evaluation scenarios;
  - writing motivation/impact sections.

## How is this different from GALILEO?

- This is **pedagogy + HCI** (curriculum, learning outcomes, inoculation theory), not an algorithmic/modeling contribution.
- Lacks empirical benchmarks/measurement; more of a position paper.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes technical methods or evaluation protocols, it likely offers more measurable progress (metrics/benchmarks) than this conceptual intervention proposal.

## Where GALILEO is weaker / needs to improve

- If GALILEO is mainly model-side, it may under-address the **user-skill / interaction literacy** angle; this paper is a useful reminder that “safe behavior” may require co-design with user-facing interventions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding an “inoculation” angle in related work: argue that some failures (e.g., sycophancy) may need user-side defenses; position GALILEO relative to pedagogical interventions.
- [ ] Add discussion/limitations: technical fixes may be insufficient because users can *prefer* disempowering interactions (per Sharma et al.); tie to evaluation of user preference vs long-term agency.
- [ ] If GALILEO has user studies or interaction patterns, consider a small experiment: does brief guided exposure to model failure modes improve calibration/avoidance?

## Quotes / details to potentially cite

- The core move: AI literacy “cannot be acquired through declarative knowledge alone” and requires guided exposure to AI failure modes (framed via inoculation theory).
- Identifies three disempowerment mechanisms (from Sharma et al.): **reality distortion**, **value judgment distortion**, **action distortion**; notes **sycophantic validation** as dominant.
- LO1 operationalization: per output, decide **accept / verify / escalate**; explicit risk (traffic-light) categorization exercises.
