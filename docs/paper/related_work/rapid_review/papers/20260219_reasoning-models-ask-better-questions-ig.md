# Do Reasoning Models Ask Better Questions? A Formal Information-Theoretic Analysis on Multi-Turn LLM Games

- Year: 2026
- Venue: NeusymBridge Workshop @ AAAI 2026 (workshop)
- Authors: Bryan Lincoln Marques De Oliveira et al.
- URL: https://arxiv.org/abs/2601.17716
- BibTeX key (if we add it): deoliveira2026_reasoning_questions_ig (suggested)
- Tags: multi-turn, information-seeking, question-asking, information-gain, 20-questions

## One-sentence takeaway

They propose an information-theoretic, multi-turn “guess my city” framework that scores each yes/no question by entropy reduction (information gain), and find that models with explicit reasoning (CoT) generally achieve higher IG per turn and solve in fewer turns, especially under partial observability.

## What problem does it solve?

- Existing question-asking / “20 questions” style benchmarks often only score the final outcome (success/steps) and do not provide a principled *turn-level* signal of question quality.
- Prior work rarely compares “reasoning traces / CoT enabled” vs “no-CoT” versions of the *same* models under a consistent evaluation protocol.

## What is the core method / protocol?

- Environment: an explicit, hierarchical hypothesis space represented as a knowledge graph (tree-like taxonomy).
  - Instantiation: “Guess My City” over a 5-level geography taxonomy: region → subregion → country → state → city.
  - Candidate set: 40 most populous cities (to reduce oracle-knowledge confounds).
- Three-agent game loop:
  - **Seeker** (the model being evaluated) asks yes/no clarification questions.
  - **Oracle** answers yes/no based on the hidden target city and decides if the Seeker has identified it.
  - **Pruner** updates the hypothesis space by eliminating inconsistent nodes after each Q/A.
- Two observability settings:
  - **Fully observable (FO):** Seeker sees the current graph state + dialogue history.
  - **Partially observable (PO):** Seeker only sees dialogue history (must implicitly track/prior over remaining hypotheses).
- Scoring: compute per-turn and cumulative **Information Gain** via Shannon entropy under a uniform prior over remaining candidates.
  - H = log2(N) where N is number of active candidates; IG = H_before − H_after after pruning.

## What are the key metrics?

- **IG per turn** (bits) and **cumulative IG** across the dialogue.
- **Turns to solution** (steps until the correct city is identified), reported across targets/runs.
- Analysis of reasoning traces (qualitative / behavioral characterization): exploration vs assertiveness in candidate-question generation.

## What are the main results?

- Models with explicit reasoning / CoT tend to:
  - achieve **higher IG per question**,
  - reach the correct target in **fewer turns**,
  - with the gap most pronounced in **partially observable** settings.
- Behavioral analysis claim:
  - smaller models “compensate” by exploring more candidate questions,
  - larger models are more assertive in selecting high-IG queries (generate candidates with higher potential IG).

## How is this similar to GALILEO?

- Both are about **multi-turn information seeking / clarification** and measuring whether an agent asks questions that reduce uncertainty.
- Uses a **structured hypothesis space** notion (explicit candidate set) which aligns with evaluation setups where GALILEO may track candidate interpretations / disambiguation states.

## How is this different from GALILEO?

- This is primarily an **evaluation framework** (plus analysis) for a stylized yes/no game over an explicit taxonomy; it does not propose a new end-to-end agent architecture for real user tasks.
- Relies on a **known, enumerable hypothesis space** (40 cities) and a pruner that can deterministically eliminate candidates; real tasks may have open-world hypotheses and fuzzier constraints.
- The “Oracle” is another LLM (not ground truth) which can introduce answering errors unless tightly controlled.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets realistic tasks, it may better capture:
  - non-binary questions,
  - richer user interaction (beyond yes/no),
  - unenumerated / open-ended hypothesis spaces.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a principled *turn-level* metric for question utility, this paper’s IG framing is a clean option.
- If GALILEO does not test partial observability (agent only has dialogue history), this PO setting is a useful stress test for “implicit belief tracking.”

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an **IG-style metric** (entropy reduction / candidate-set reduction) to GALILEO’s clarification evaluation when a hypothesis set can be enumerated (even approximately).
- [ ] Include an explicit **partial observability** condition (hide internal state / candidate set from the question-asking model) to test whether question quality degrades.
- [ ] In related work, cite this as an example of **turn-level scoring** for question asking (vs only final success metrics).

## Quotes / details to potentially cite

- “We adopt IG as the main metric, grounded in Shannon entropy, to assess query effectiveness at each turn and cumulatively.” (abstract)
- Framework: triad of agents (Seeker/Oracle/Pruner) operating over a hierarchical knowledge graph; instantiation: 40-city Guess-My-City with region→subregion→country→state→city taxonomy. (Sec. 3–4)
