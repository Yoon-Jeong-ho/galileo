# MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers

- Year: 2025
- Venue: arXiv
- Authors: Xuanjun Zong; Zhiqi Shen; Lei Wang; Yunshi Lan; Chao Yang
- URL: https://arxiv.org/abs/2512.15163
- BibTeX key (if we add it): Zong2025MCPSafetyBench
- Tags: mcp, tool-use, agent-safety, benchmark, multi-turn, multi-server

## One-sentence takeaway

MCP-SafetyBench provides an execution-based, multi-turn benchmark on *real MCP servers* with a 20-type MCP-specific attack taxonomy, showing today’s LLM agents have substantial vulnerability that worsens with longer tool-use horizons and more server interactions.

## What problem does it solve?

- Existing “MCP safety” / tool-attack benchmarks often (per authors) (i) cover only a narrow subset of attacks, (ii) are not built on realistic MCP server integrations, and/or (iii) do not capture multi-turn, multi-server workflows where attacks can occur mid-trajectory.
- Need an evaluation that measures both **task success** and **attack success** under controlled attack injections, in realistic MCP settings.

## What is the core method / protocol?

- Build on MCP-Universe tasks and transform each into an attack-instrumented case across 5 domains:
  - browser automation, financial analysis, location navigation, repository management, web search
- Define a unified taxonomy of **20 MCP attack types** spanning three sides:
  - MCP server-side (e.g., tool poisoning variants, function overlapping/shadowing, function-return injection, rug pull)
  - MCP host-side (intent injection, data tampering, identity spoofing, replay injection)
  - user-side (malicious code execution, credential theft, retrieval-agent deception, excessive privilege misuse, etc.)
- Construction pipeline (high level): select baseline task → instantiate exactly one attack modification at server/host/user side → package with a manifest + automated evaluators.
- Evaluation is execution-based and outputs a **dual label**:
  - task outcome (pass/fail wrt goal)
  - attack outcome (attack success/failure wrt attack objective)

## What are the key metrics?

- **TSR (Task Success Rate)**: % tasks where the user goal is achieved.
- **ASR (Attack Success Rate)**: % tasks where the attack objective succeeds (higher = worse safety).
- (Also discussed) a “defense success rate” notion as 1 - ASR, and the trade-off trend between utility (TSR) and robustness.

## What are the main results?

- Benchmark size: **245** cases total across 5 domains (30–56 per domain).
- Attacks are mostly server-side in their design (reported ~75% server-side; ~12% host; ~13% user) and roughly split disruption vs stealth.
- Across 13 evaluated models, **no model is immune**; overall ASR ranges roughly from ~30% (best in their table) to ~48% (worst), depending on model.
- Host-side attacks are especially severe (authors report very high average success for host-side; identity-spoofing/replay/intent-style classes look particularly damaging in their breakdown).
- They report a negative correlation between “doing tasks well” and “defending well” (safety–utility trade-off), and that vulnerabilities compound with longer horizons / more server interactions.
- Prompt-only mitigation (“Safety Prompt”) provides only small overall improvement and is attack-type dependent (helps for some, hurts for others).

## How is this similar to GALILEO?

- Shares the core theme that **multi-turn agentic behavior** can drift/fail under adversarial pressure during *trajectories*, not just at a single turn.
- Explicitly emphasizes evaluating agents in **realistic tool ecosystems** (MCP) rather than purely synthetic prompts.
- Dual-outcome evaluation (utility vs attacker objective) is conceptually aligned with GALILEO’s need to avoid “safe by refusing everything” artifacts.

## How is this different from GALILEO?

- This paper is primarily about **tool/MCP attack surfaces** (server/host/user side) rather than *social pressure / belief drift / sycophancy* per se.
- Evaluation endpoints are **task completion + attack detectors** in tool workflows; GALILEO (as positioned in our related-work shortlist) is more about *conversational multi-turn robustness* (flip dynamics, recovery, pressure vs evidence).
- Their tasks span operational domains (repo management, navigation, finance), whereas GALILEO’s core artifacts are likely dialogue-focused robustness metrics and controls.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has tight controls for **pressure-only vs evidence-driven revision** and explicit **recovery-after-failure** measurements, that is a clearer causal story than generic “task failed/attack succeeded” in heterogeneous tool tasks.
- GALILEO can present more interpretable *behavioral trajectories* (flip types, hazards, recovery) rather than many heterogeneous, domain-specific evaluators.

## Where GALILEO is weaker / needs to improve

- If GALILEO does not yet include **tool-ecosystem realism** (multi-server/tool-call traces), this paper is a reminder that real deployments face additional failure modes:
  - schema/parameter poisoning, tool name collisions, function-return prompt injection, host message tampering
- If GALILEO reporting does not include an explicit **utility vs safety** decomposition, MCP-SafetyBench provides precedent for separating task success from attack success.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite MCP-SafetyBench as “real MCP servers + multi-turn + multi-server” benchmark; use it to motivate that *trajectory-length* and *cross-system interactions* amplify risk.
- [ ] Consider adding (even as a small ablation/appendix) a **tool-mediated pressure channel** (e.g., retrieved/tool-return prompt injection) to show GALILEO’s controls extend beyond plain dialogue.
- [ ] Consider adopting the *dual metric* framing explicitly: report a “utility” axis and a “robustness/attack-resistance” axis to avoid degenerate baselines.

## Quotes / details to potentially cite

- “MCP-SafetyBench … built on real MCP servers … supports realistic multi-turn evaluation across five domains …” (abstract)
- “unified taxonomy of 20 MCP attack types spanning server, host, and user sides” (abstract)
- Benchmark stats: 245 cases; domains and counts (Table 3 in HTML version).
- Evaluation outputs dual labels: task success and attack success (Section 3.5).
