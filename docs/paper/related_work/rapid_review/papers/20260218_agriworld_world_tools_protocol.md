# AgriWorld: A World–Tools–Protocol Framework for Verifiable Agricultural Reasoning with Code-Executing LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Zhixing Zhang; Hao Liu; Qinhan Lv; Jing Yang; Kaitong Cai; Keze Wang
- URL: https://arxiv.org/abs/2602.15325
- BibTeX key (if we add it): AgriWorld2026
- Tags: agents, tool-use, code-execution, verifiable-eval, world-tools-protocol, agriculture

## One-sentence takeaway

An agriculture-focused code-executing LLM agent (execute–observe–refine) operating in a unified “World–Tools–Protocol” environment beats text-only and direct tool-use baselines on a verifiable, tool-grounded QA benchmark.

## What problem does it solve?

- Pure text LLMs can’t reliably answer agronomic questions that require **spatiotemporal numeric computation** (remote sensing time series, parcel geometry joins, soil/weather grids, simulation).
- Domain foundation models can forecast/monitor, but lack **interactive, language-driven reasoning** and “what-if” analysis.
- Existing agent evals often score final text; this paper pushes for **executable checkers** to make evaluation verifiable and to diagnose errors (CRS misalignment, windowing/unit mistakes).

## What is the core method / protocol?

- **World–Tools–Protocol abstraction**:
  - **World (AgriWorld):** a Python execution environment exposing unified APIs for core agronomic operations:
    - geospatial queries over parcels/regions
    - remote-sensing time-series analytics + anomaly statistics
    - crop growth simulation for counterfactuals
    - task predictors (yield / stress / disease risk)
  - **Tools:** return inspectable artifacts (tables/plots/masks) for auditability.
  - **Protocol:** standardizes task specification + evaluation with **deterministic reference code** and **executable checker functions** when possible.
- **Agro-Reflective agent:** a multi-turn agent that alternates **write code → execute → observe artifacts/errors → refine** (execute–observe–refine). Emphasis is on execution feedback as a first-class signal (debugging CRS/time-window/unit issues).
- **AgroBench:** benchmark with scalable data generation for diverse agricultural QA: lookups, forecasting, anomaly detection, and counterfactual “what-if”.

## What are the key metrics?

- Paper frames evaluation as verifiable via **executable checkers** (i.e., correctness grounded in reference programs), rather than only text matching.
- Reports performance versus baselines like **text-only** and **direct tool-use** (single-shot) agents (exact metric names not captured from the truncated HTML; likely task accuracy / pass rate via checkers).

## What are the main results?

- **Agro-Reflective** (multi-turn execute–observe–refine) outperforms:
  - **text-only** LLM baselines, and
  - **direct tool-use** baselines (one-shot tool calling without iterative execution-driven reflection),
  on AgroBench’s tool-grounded tasks.
- Qualitative/diagnostic claim: iterative execution feedback improves reliability on failure modes common in agriculture (spatial alignment, temporal windows, units).

## How is this similar to GALILEO?

- Strong conceptual overlap with GALILEO’s emphasis on:
  - **multi-turn protocols** (iterated interaction rather than one-shot answers),
  - **auditability / intermediate artifacts**, and
  - **verifiable evaluation** (executable checks, clearer failure attribution than pure text judging).
- Their “Protocol” component is philosophically aligned with building **paired, checkable variants** and making evaluation reproducible.

## How is this different from GALILEO?

- Domain: agriculture + geospatial/time-series/simulation toolchains (not social pressure / belief drift / persuasion).
- Core failure modes: numeric/spatiotemporal correctness (CRS, windowing, units) rather than social influence, stance drift, or conversational pressure.
- Agent loop is explicitly **code-execution-centered**; GALILEO may evaluate language behavior with fewer domain-specific tools.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets social-pressure robustness, it may offer a **cleaner causal isolation** of “pressure-only” effects vs legitimate evidence updates, whereas AgriWorld’s setting is primarily “do the computation correctly.”

## Where GALILEO is weaker / needs to improve

- GALILEO could benefit from adopting an explicit **World–Tools–Protocol** story:
  - “World” = environment state + tools
  - “Tools” = typed operations returning auditable artifacts
  - “Protocol” = standardized task specs + executable checkers
- If GALILEO currently relies heavily on textual judges, this paper is another datapoint pushing toward **more executable, programmatic evaluation** where feasible.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a short framing section: “GALILEO as World–Tools–Protocol” (even if our tools are lightweight), to clarify what is being held fixed and what changes across conditions.
- [ ] Where possible, add **executable checker** versions of a subset of GALILEO tasks (unit tests / constraints) to reduce judge dependence and enable finer-grained failure attribution.
- [ ] In writing, borrow the pitch: *interactive assistants need execution + auditability; final-text scoring is insufficient for diagnosis.*

## Quotes / details to potentially cite

- “World–Tools–Protocol abstraction” for executable, auditable scientific assistants.
- Agro-Reflective: execute–observe–refine loop; motivation that small spatial/temporal/unit mistakes can invalidate conclusions.
- Verifiable evaluation via deterministic reference programs and executable checker functions (when admissible).