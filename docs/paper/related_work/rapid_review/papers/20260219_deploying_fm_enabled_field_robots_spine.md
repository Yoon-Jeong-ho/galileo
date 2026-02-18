# Deploying Foundation Model-Enabled Air and Ground Robots in the Field: Challenges and Opportunities

- Year: 2025
- Venue: IEEE ICRA Workshop on Field Robotics (accepted; arXiv preprint)
- Authors: Zachary Ravichandran, Fernando Cladera, Jason Hughes, Varun Murali, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar
- URL: https://arxiv.org/abs/2505.09477
- BibTeX key (if we add it): ravichandran2025deploying
- Tags: robotics, deployment, foundation-models, llm-planning, closed-loop-validation, distillation, uav, ugv, communication

## One-sentence takeaway

Field deployment of an LLM-in-the-loop autonomy framework (SPINE) shows km-scale UGV missions with closed-loop plan validation, and sketches early distillation results enabling fully on-device language-driven UAV planning.

## What problem does it solve?

- Most FM/LLM-enabled robot planners assume a closed world: complete prior map, structured environment, short-horizon tasks.
- Field robots face partial/unknown maps, unstructured terrain, online discovery (active mapping), sensor surprises, and limited compute / intermittent comms.
- Need a practical architecture that (a) can use LLM reasoning for under-specified missions, but (b) stays safe/grounded under online map updates and failures.

## What is the core method / protocol?

- Deploy SPINE: an autonomy framework with two key modules:
  - Plan generation: a (frontier) LLM produces a task sequence over a robot behavior API (e.g., goto, explore_region, extend_map, inspect), conditioned on a semantic map/graph and mission text; replans iteratively as map updates arrive.
  - Plan validation: checks each proposed behavior against constraints (syntax + “reachability” + “explorable” constraints) to ground the plan in what is currently feasible/safe; returns natural-language feedback to the LLM for repair.
- System is integrated into real autonomy stacks:
  - UGV (Clearpath Jackal) with LiDAR odometry + freespace/trajectory planning; semantic mapping.
  - UAV (Falcon 4) with PX4 + waypointing behaviors; semantic map + geofence.
- Communication: server-based LLM use supported via mesh networking (radios + upstream internet), motivating edge models.
- On-device planning (preliminary): distill GPT-4o “expert planner” into a small language model (LoRA finetune of Llama-3.2 3B) using collected (specification, semantic graph) -> (observation/action plan) data.

## What are the key metrics?

- UGV field mission outcomes across 14 missions: overall mission success count; failure modes (communication loss, odometry drift; some obstacle detection issues requiring brief teleop).
- Scale: missions spanning ~100 m to ~1 km; “kilometer-scale” LLM-enabled planning in unstructured environments.
- Distilled planning quality: plan-correctness rate on 11 spec/graph test cases comparing GPT-4o vs distilled model vs off-the-shelf base model.

## What are the main results?

- UGV deployments: SPINE completed 12/14 missions; failures attributed primarily to comms loss and odometry drift; some perception failures were recoverable with brief manual takeover.
- Emphasizes that online plan validation materially improves success as environments become more unknown (they highlight a notable drop without validation).
- UAV (on-device) demo: with an extended semantic map, the distilled onboard model successfully executed 3/4 natural-language missions “on first try”; the miss was a semantic confusion between two similar locations (north vs south parking lot).
- Distillation evaluation: GPT-4o achieved 100% (11/11) on their short-horizon planning test; distilled Llama-3.2 3B reached 72.7% (8/11) vs off-the-shelf Llama-3.2 3B substantially worse (table referenced; key point: large gain from distillation but still a gap to the frontier model).

## How is this similar to GALILEO?

- “Closed-loop” robustness framing: detect/handle model errors (hallucinations) via structured validation and feedback, rather than one-shot prompting.
- Explicitly addresses real-world failure modes and non-ideal conditions (partial observability, online updates, comms constraints), which aligns with robustness-under-pressure motivations.
- Distillation as a strategy to reduce dependence on fragile infrastructure (server access), akin to reducing reliance on idealized settings.

## How is this different from GALILEO?

- Domain: robotics autonomy and semantic mapping, not multi-turn dialogue robustness / sycophancy / belief revision evaluation per se.
- Their “robustness” lever is mostly *task/plan validation* and autonomy-stack constraints; less about conversational dynamics, persuasion, or belief drift across dialogue rounds.
- Evaluation is field-mission success and planning correctness, not standardized multi-turn behavioral benchmarks.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides clearer multi-turn robustness definitions/metrics (e.g., drift, stability, adversarial dialogue), it can be more precise than qualitative “lessons learned” from field deployments.
- GALILEO likely has tighter experimental control and ablations targeted to multi-round interaction pathologies.

## Where GALILEO is weaker / needs to improve

- Concrete grounding/verification: SPINE-style validation shows a practical pattern (constraint checks + feedback loop) for keeping a powerful model aligned with a changing world state.
- Realistic “systems constraints” thinking (comms, SWaP/edge compute) is often missing in purely conversational robustness work; this paper is a reminder to consider deployment constraints.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related-work, consider citing SPINE as an example of *closed-loop validation* as a robustness mechanism (model proposes; validator rejects with actionable feedback; model repairs), even if in robotics.
- [ ] Map SPINE’s validator loop to GALILEO’s setting: define an analogous “dialogue plan validator” (e.g., constraint-based checks for consistency, evidence tracking, or disallowed persuasion tactics) that produces *natural-language* repair feedback.
- [ ] If GALILEO discusses distillation/compact models, mention this as an existence proof of distilling planning behavior for edge constraints; note the observed gap and failure patterns (multi-iteration/map-update struggles).

## Quotes / details to potentially cite

- Claim (paraphrase): first demonstration of large-scale LLM-enabled robot planning in unstructured environments with kilometer-scale missions.
- SPINE structure: plan generation over behavior API + plan validation enforcing syntax/reachability/explorable constraints, with NL feedback to repair hallucinated plans.
- Distillation headline: LoRA finetune Llama-3.2 3B to mimic GPT-4o planner; improves plan correctness materially vs base SLM but still lags frontier model; struggles with multi-iteration planning and map-update responses.
