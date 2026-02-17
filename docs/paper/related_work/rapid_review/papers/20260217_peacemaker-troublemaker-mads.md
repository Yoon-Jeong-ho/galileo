# Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate

- Year: 2025
- Venue: arXiv
- Authors: Binwei Yao; Chao Shang; Wanyu Du; Jianfeng He; Ruixue Lian; Yi Zhang; Hang Su; Sandesh Swamy; Yanjun Qi
- URL: https://arxiv.org/abs/2509.23055
- BibTeX key (if we add it): yao2025peacemaker
- Tags: sycophancy, multi-agent, debate, disagreement-collapse, robustness

## One-sentence takeaway

In multi-agent debate systems, inter-agent sycophancy drives **disagreement collapse** (premature convergence to wrong consensus), and persona control along a “troublemaker↔peacemaker” spectrum can partially mitigate it—especially in heterogeneous debates.

## What problem does it solve?

- Prior sycophancy work mostly studies **user↔LLM** interactions; this paper targets **agent↔agent sycophancy** inside multi-agent debating systems (MADS).
- Identifies how sycophancy can break the core promise of debate: using disagreement to converge to correct answers.

## What is the core method / protocol?

- Defines **sycophancy in MADS** as: excessive alignment with other agents that prioritizes harmony over role objectives.
- Studies two MADS architectures:
  - **Decentralized**: Society-of-Minds (SoM)-style peer debate (no judge).
  - **Centralized**: debaters + a **judge** (judge chooses an agent’s answer).
- Benchmarks/tasks:
  - CommonsenseQA (full validation set).
  - MMLU-Pro (1,000 sampled).
- Backbone models:
  - Qwen3-32B
  - Llama 3.3-70B Instruct
- Key experimental levers:
  - 2-agent and 3-agent debates.
  - Homogeneous vs heterogeneous model compositions.
  - **Sycophancy-control personas**: discrete “sycophancy level” \(\lambda \in \{1..8\}\) set via system prompts, from **troublemaker** (low \(\lambda\), independent/disagreeing) to **peacemaker** (high \(\lambda\), harmony/agreeing). Grid search over persona combinations.

## What are the key metrics?

Debate status taxonomy (per round):
- **PA** (Positive Agreement): unanimous correct.
- **NA** (Negative Agreement): unanimous incorrect.
- **PD** (Positive Disagreement): disagreement exists and at least one agent is correct.
- **ND** (Negative Disagreement): disagreement exists and all agents are incorrect.

Metrics:
- **DCR (Disagreement Collapse Rate)**: among cases starting in PD at round 0, fraction that fail to end in PA.
- **NAR (Negative Agreement Rate)**: agent-level; how often an agent abandons a correct position during disagreement.
- **SS (Sycophancy Score)**: 0–100 score judging whether the agent’s next-turn response is “independent reasoning” vs “echoing others” (computed using an LLM-based evaluator prompt).

## What are the main results?

- Multi-agent debate does **not** reliably beat single-agent baselines; when it helps, gains are often modest given extra compute.
- **Disagreement collapse is common**, especially in decentralized settings.
  - Example reported: homogeneous Llama3.3-70B 2-agent CommonsenseQA has DCR up to **86.36%**.
  - Centralized (judge) setups reduce DCR substantially (example: CommonsenseQA Qwen–Qwen DCR drops from **81.71%** decentralized to **41.27%** centralized).
- Strong evidence that collapse is sycophancy-linked:
  - Debater **NAR vs SS** correlation: Pearson **r = 0.902**.
  - Judge **DCR vs SS** correlation: Pearson **r = 0.639**.
- Persona control findings:
  - Worst configurations are typically **high-high** (both peacemakers) → higher collapse and lower accuracy.
  - Best configurations are often **mixed** (peacemaker + troublemaker) rather than all-troublemakers, suggesting a “steerability vs independence” trade.
  - Persona control matters more in **heterogeneous debates**; they report a wider accuracy swing (e.g., in Qwen–Llama 2-agent CommonsenseQA, ~78.95% to ~82.06%, with best at troublemaker–troublemaker).
- Judge persona appears relatively **insensitive** (accuracy relatively stable across judge sycophancy levels in their tested setups).

## How is this similar to GALILEO?

- Same failure mode family: **social-pressure-induced drift** (here: pressure from peers/other agents) leading to collapse before reaching the truth.
- Introduces **trajectory/state** thinking (PD→PA vs PD→NA/ND) that maps well to GALILEO’s multi-turn stability framing.

## How is this different from GALILEO?

- This is **multi-agent debate** (agent–agent), not single model resisting a user’s pressure.
- Primary outcome is debate-level correctness / collapse; it does not target **belief revision vs drift controls**, nor explicit **recovery-after-flip** trajectories.
- Uses an LLM-judge (“Blind Reasoning” evaluator) for SS; GALILEO may want more auditable / non-LLM measures.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO separates **evidence-driven revision** from **pressure-driven drift**, that is cleaner than debate settings where “new information” and “social influence” are entangled.
- If GALILEO reports survival/time-to-failure and recovery dynamics, it adds a dimension this paper largely lacks.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently focuses on single-agent settings only, this paper is a reminder that **agentic/multi-agent deployments** can introduce additional sycophancy channels and failure modes.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “multi-agent implication” paragraph: inter-agent sycophancy can collapse deliberation, so robustness should be assessed under **peer pressure**, not only user pressure.
- [ ] Consider importing **PD/PA/ND/NA** state taxonomy as an interpretable reporting layer (or define an isomorphic state machine for GALILEO’s turn-by-turn outcomes).
- [ ] If we use any LLM-based evaluator, explicitly report robustness to evaluator choice (since this paper’s SS relies on an LLM prompt).

## Quotes / details to potentially cite

- Core claim (abstract): sycophancy in MADS “amplifies disagreement collapse before reaching a correct conclusion … yields lower accuracy than single-agent baselines … arises from distinct debater-driven and judge-driven failure modes.”
- Metric definitions:
  - DCR: fraction of PD0 cases that do not become PA at final round.
  - SS: 0–100 score for “independent reasoning vs echoing others.”
- Correlations linking sycophancy to collapse:
  - Debater NAR vs SS: r=0.902.
  - Judge DCR vs SS: r=0.639.
