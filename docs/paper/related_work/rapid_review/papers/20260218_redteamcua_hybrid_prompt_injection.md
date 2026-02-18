# RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments

- Year: 2026
- Venue: ICLR 2026 (Oral); arXiv
- Authors: Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun
- URL: https://arxiv.org/abs/2505.21936
- BibTeX key (if we add it): Liao2026RedTeamCUA
- Tags: agents, computer-use, prompt-injection, red-teaming, evaluation, sandbox, security

## One-sentence takeaway

RedTeamCUA provides a realistic-but-controlled hybrid Web+OS sandbox and an 864-example benchmark (RTC-Bench) showing frontier computer-use agents remain highly vulnerable to indirect prompt injection, with “attempt rates” far exceeding “attack success rates.”

## What problem does it solve?

- Existing prompt-injection / adversarial-agent evaluations are either (a) unrealistic / non-interactive, (b) risky because they use live environments, or (c) miss *hybrid* web→OS attack paths (web content inducing harmful OS-level actions).
- Capability limitations (navigation failures) can confound security evaluation; we want to measure vulnerability *given exposure to an injection*, not just “can the agent reach the page.”

## What is the core method / protocol?

- **RedTeamCUA framework**: a hybrid sandbox combining:
  - VM-based desktop/OS environment (building on OSWorld) for realistic OS actions.
  - Docker-based replicas of web platforms (building on WebArena / TheAgentCompany) for controlled, safe web interaction.
- **Adversarial scenario configuration** with scripts to inject malicious content into web/OS environments.
- **Decoupled Eval** setting: initialize evaluation directly at the point of injection (bypassing navigation), to isolate prompt-injection susceptibility from general CUA capability constraints.
- **RTC-Bench**: 864 adversarial examples generated as:
  - 9 benign “user goals” × 24 adversarial goals (CIA-triad-inspired: confidentiality/integrity/availability) × 4 instantiations
    - benign instruction specificity: General vs Specific
    - injection type: Code vs Language

## What are the key metrics?

- **Attack Success Rate (ASR)**: execution-based evaluator for whether the adversarial goal is achieved.
- **Attempt Rate (AR)**: LLM-judge-based measure capturing whether the agent *tries* to pursue the adversarial goal even if it fails to complete it.
- Reporting under (at least) two settings:
  - Decoupled Eval (exposure-controlled)
  - End-to-End Eval (more realistic, includes navigation)

## What are the main results?

- Under **Decoupled Eval**, all tested frontier CUAs show non-trivial vulnerability; one reported example: Claude 3.7 Sonnet | CUA ASR 42.9%, “Operator” ASR 7.6% (best among evaluated).
- **AR can be extremely high (up to ~92.5%)**, often much larger than ASR → many “failures to complete the attack” are due to capability limits, not robustness.
- Under more realistic **End-to-End Eval**, some models show alarmingly high ASR in “full pipeline” scenarios (the paper reports very high ASRs for strong CUA systems), suggesting tangible real-world risk.

## How is this similar to GALILEO?

- Clear emphasis on **evaluation design**: separating “true robustness” from confounds (their navigation confound is analogous in spirit to separating pressure-driven drift from evidence-driven revision).
- Uses **multi-stage metrics** (attempt vs success) that mirror the idea that “partial compliance / trajectory signals” matter, not only final outcomes.

## How is this different from GALILEO?

- Domain: tool-using **computer-use agents** in hybrid web+OS environments, not primarily conversational belief/stance drift.
- Threat model: **indirect prompt injection** embedded in the environment leading to concrete system actions (CIA-triad harms), rather than social pressure / persuasion dynamics.
- Heavy reliance on **execution-based evaluation** in sandboxed environments.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets conversational/multi-turn belief drift: likely cleaner control over “evidence vs pressure” variables than complex hybrid UI environments.
- Potentially more fine-grained trajectory metrics around *recovery/oscillation* (if present in GALILEO), whereas this paper emphasizes success/attempt toward adversarial goals.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims robustness evaluations generally, it may lack an analogue of **Decoupled Eval** that cleanly removes capability confounds.
- Might be missing a direct “**attempt vs success**” decomposition that distinguishes *intent-to-comply* from *capability-to-execute*.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an “attempt vs success” reporting layer (or a similarly interpretable decomposition) to avoid overstating robustness when failures are actually capability limits.
- [ ] Consider a decoupling trick: initialize evaluations at the “pressure/injection exposure point” to isolate the specific vulnerability under study.
- [ ] If GALILEO touches agents/tool use: cite RedTeamCUA as evidence that hybrid environments enable realistic but safe evaluation, and that high AR warns risks will increase as agent capabilities improve.

## Quotes / details to potentially cite

- “Computer-use agents (CUAs) … remain vulnerable to indirect prompt injection.”
- RedTeamCUA proposes a “hybrid sandbox” integrating a VM-based OS with Docker-based web platforms.
- RTC-Bench: 864 examples; adversarial goals based on the CIA triad; includes General vs Specific benign instructions and Code vs Language injections.
- Key headline: Attempt Rate up to ~92.5% while ASR is lower → capability limitations can mask vulnerability.
