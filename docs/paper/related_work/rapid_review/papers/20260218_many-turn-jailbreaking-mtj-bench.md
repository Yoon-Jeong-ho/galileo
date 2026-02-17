# Many-Turn Jailbreaking

- Year: 2025
- Venue: arXiv
- Authors: Xianjun Yang; Liqiang Xiao; Shiyang Li; Faisal Ladhak; Hyokun Yun; Linda Ruth Petzold; Yi Xu; William Yang Wang
- URL: https://arxiv.org/abs/2508.06755
- BibTeX key (if we add it): yang2025manyturn
- Tags: multi-turn, jailbreak, benchmark, safety

## One-sentence takeaway

MTJ-Bench argues that **jailbreak success should be evaluated across the full follow-up conversation**, and introduces a benchmark for measuring how jailbreaks persist/propagate across multiple turns.

## What problem does it solve?

- Most jailbreak evaluations are **single-turn**: they test whether a model can be induced to answer one unsafe query once.
- In real usage, once a model “gives in” it may:
  - continue providing additional harmful detail under **follow-up clarification**, and/or
  - remain in a compromised state and answer **subsequent (even irrelevant) queries** unsafely.
- This paper frames that as a distinct threat model: **multi-turn jailbreaking**.

## What is the core method / protocol?

- Define a multi-turn setting where, after an initial jailbreak, the model is further probed over subsequent turns.
- Construct **MTJ-Bench** to benchmark this behavior on a mix of open- and closed-source LLMs.
- (From the abstract) emphasis is on persistence beyond first-turn: “continuously tested on more than the first-turn conversation or a single target query.”

## What are the key metrics?

- Not clearly extractable from the abstract alone.
- Likely outcome variables (implied): whether unsafe behavior persists across turns, and whether it generalizes to additional follow-ups / irrelevant questions.

## What are the main results?

- Not clearly extractable from the abstract alone.
- Main contribution (per abstract): a benchmark + “novel insights” showing multi-turn jailbreak is a serious vulnerability.

## How is this similar to GALILEO?

- Shares the core motivation that **single-turn pass/fail is insufficient** for safety evaluation; dynamics across turns matter.
- Reinforces the importance of **trajectory-level** evaluation (what happens after an initial failure event).

## How is this different from GALILEO?

- MTJ-Bench focuses on **unsafe-content jailbreaking** (policy/safety violation elicitation), while GALILEO centers more on **multi-turn belief/stance dynamics under pressure** (drift vs evidence-driven revision, recovery).
- MTJ-Bench’s threat model includes “compromised state” persistence; GALILEO’s framing typically distinguishes pressure vs evidence and cares about **recovery / return-to-truth**.

## Where GALILEO is stronger / cleaner (if true)

- GALILEO can position itself as providing **cleaner causal controls** (pressure vs evidence; recovery interventions) rather than just “continued querying after jailbreak.”
- GALILEO’s metrics can be reported as **time-to-event / recovery trajectories**, which may be more diagnostic than aggregate multi-turn success rates.

## Where GALILEO is weaker / needs to improve

- If MTJ-Bench includes a broad suite of realistic multi-turn adversarial follow-ups, GALILEO may need to ensure it covers:
  - follow-up clarification chains, and
  - cross-topic “state contamination” style probes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph explicitly distinguishing **multi-turn jailbreak persistence** vs **multi-turn belief drift/recovery**, while citing MTJ-Bench as evidence the community recognizes multi-turn evaluation as necessary.
- [ ] Consider adding a small “state persistence / contamination” probe: after a pressure-induced failure, test whether the model is more likely to fail on a later, loosely-related probe.

## Quotes / details to potentially cite

- Abstract: “we propose exploring multi-turn jailbreaking, in which the jailbroken LLMs are continuously tested on more than the first-turn conversation or a single target query.”
- Abstract: multi-turn is a “more serious threat” because users ask follow-ups and because initial jailbreak may cause unsafe responses to additional irrelevant questions.
