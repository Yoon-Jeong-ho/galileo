# Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents

- Year: 2026
- Venue: arXiv
- Authors: Kaiyu Zhou; Yongsen Zheng; Yicheng He; Meng Xue; Xueluan Gong; Yuji Wang; Kwok-Yan Lam
- URL: https://arxiv.org/abs/2601.10955
- BibTeX key (if we add it): zhou2026beyond
- Tags: agents, tool-use, mcp, security, economic-dos, multi-turn

## One-sentence takeaway

A protocol-compatible (MCP) tool server can be made *stealthily* adversarial by only changing text-visible fields/return templates, inducing long multi-turn tool-calling loops that massively amplify tokens/cost/energy while still completing the task correctly.

## What problem does it solve?

- Identifies and operationalizes the **agent–tool interaction loop** as a DoS / resource-exhaustion attack surface that is not well covered by prior single-turn “make the model output long text” attacks.
- Shows how an attacker can achieve **economic DoS** (billing/cost), and also practical **serving degradation** (KV cache occupancy, throughput reduction) *without breaking task correctness*.

## What is the core method / protocol?

- Threat model: attacker controls (or compromises) a **tool server** used by an LLM agent; the server remains **protocol-compatible** (MCP-compatible).
- Key idea: keep function signatures and final payload intact, but edit:
  - **text-visible fields** (e.g., descriptions / notices returned to the agent)
  - a **template-governed return policy**
  to nudge the agent into **repeated, verbose tool-calling sequences** (multi-turn trajectories).
- Optimization: uses **Monte Carlo Tree Search (MCTS)** to search for server-side text/template modifications that maximize resource amplification while preserving task success.

## What are the key metrics?

- Trajectory length / total tokens over multi-turn agent-tool interaction (reported to exceed **60k tokens**, sometimes **>90k**).
- Cost amplification factor vs benign tool server (up to **658×**).
- Energy increase (reported **100–560×** range).
- Serving resource impact: **GPU KV cache occupancy** increase (from <1% to **35–74%**) and **throughput reduction** (~50% co-running throughput).

## What are the main results?

- On ToolBench and BFCL across 6 LLMs, the attack reliably turns normal tool tasks into extremely long tool-calling trajectories while keeping the final answer correct.
- Conventional checks that validate only the final answer (or rely on protocol compliance) fail to detect/stop the attack.

## How is this similar to GALILEO?

- Shares a **multi-turn** perspective: the relevant unit of analysis is an *interaction trajectory*, not a single response.
- Reinforces that evaluations/defenses should consider **process-level** signals (turn-by-turn behavior, cumulative budget/compute), not only end-task success.

## How is this different from GALILEO?

- Focus is **economic/computational DoS via tool layer**, not (primarily) social pressure, belief drift, or persuasion dynamics.
- The adversary is a **tool server** (MCP-compatible) manipulating tool-channel text/policies, rather than a user prompt, debate partner, or RAG context.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled protocols to disentangle *helpful revision* vs *harmful drift* (and/or recovery), this paper is orthogonal: it largely treats “task completed correctly” as fixed and optimizes compute/cost blow-up.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not report **trajectory-level cost / token / compute budgets**, this paper suggests reviewers may increasingly expect such measurements for agentic systems.
- Potential gap: robustness discussion may need to explicitly include **tool-channel adversaries** (malicious tool servers, protocol-compliant yet adversarial tool output text).

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or at least mention) **trajectory budget metrics**: total tokens, #tool calls, wall-clock, and optionally a cost proxy.
- [ ] Consider a “benign tool vs adversarial tool text” ablation (even if synthetic) to demonstrate sensitivity to tool-channel instructions.
- [ ] In related work, cite as evidence that **agentic safety** must monitor the *entire process* (multi-turn), not only final correctness.

## Quotes / details to potentially cite

- Abstract: “*tool calling layer as a first-class security frontier* … shift from validating final answers to monitoring the economic and computational cost of the entire agentic process.”
- Abstract: “*trajectories exceeding 60,000 tokens* … *inflates costs by up to 658x* … *energy by 100-560x*.”
- Abstract: “*GPU KV cache occupancy from <1% to 35-74%* and cuts co-running throughput by approximately *50%*.”
