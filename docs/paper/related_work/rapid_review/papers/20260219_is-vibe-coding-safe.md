# Is Vibe Coding Safe? Benchmarking Vulnerability of Agent-Generated Code in Real-World Tasks

- Year: 2025
- Venue: arXiv
- Authors: Songwen Zhao, Danqing Wang, Kexun Zhang, Jiaxuan Luo, Zhuo Li, Lei Li
- URL: https://arxiv.org/abs/2512.03262
- BibTeX key (if we add it): zhao2025susvibes
- Tags: agents, code, security, robustness, benchmark

## One-sentence takeaway

SusVibes is a repo-level, multi-turn coding-agent benchmark showing that even when agents often produce functionally-correct patches, the vast majority of those patches remain security-vulnerable, and naive prompt-based mitigations trade away functionality.

## What problem does it solve?

- Existing secure-code benchmarks often evaluate single-turn, small-context code generation (single file/function) and do not match “vibe coding” settings where an *agent* iteratively edits a *repository* with environment feedback.
- The paper targets a practical risk: production adoption of agent-generated code by users who may not rigorously security-review patches.

## What is the core method / protocol?

- Construct SusVibes: 200 feature-request tasks derived from real-world vulnerability-fix commits in OSS repos.
- Each task includes:
  - A repository context (multi-file, large project).
  - A feature request (issue-style description) created after *masking* an existing feature implementation.
  - An execution environment.
  - Two human-written test suites:
    - Functional correctness tests.
    - Security tests targeting the vulnerability class.
- Mining/curation pipeline (high level):
  - Mine vulnerability-fix commits and revert to pre-fix versions.
  - Extract/attach security tests introduced/modified in fixes and pair with existing functional tests.
  - Mask feature implementation code to create a “missing feature” state.
  - Generate/verify an issue-style feature request so agents must re-implement the feature.
- Evaluate many combinations of agent frameworks + frontier LLMs on the same tasks; score patches by whether they pass functional tests and security tests.

## What are the key metrics?

- Functional correctness: % tasks where the agent patch passes functionality unit tests.
- Security: % tasks where the patch passes security unit tests.
- “Both”: % tasks that are simultaneously functionally correct and secure (implied by the setup; emphasized as the practical bar).

## What are the main results?

- Agents can be functionally successful while still insecure:
  - Reported example: SWE-Agent + Claude 4 Sonnet achieves ~61% functional pass rate, but only ~10.5% secure solutions.
  - The paper claims that >80% of functionally-correct solutions still contain vulnerabilities (i.e., fail security tests).
- Prompt-based “preliminary security strategies” (e.g., generic security guidance, prompting the model to identify CWE risk, or giving oracle CWE hints) can improve security but dramatically reduce functional correctness (reported drop on the order of ~77 percentage points), reducing the number of solutions that are both correct and secure.

## How is this similar to GALILEO?

- Shared theme: *multi-turn robustness* under realistic interaction protocols (here: iterative repo editing with environment feedback; in GALILEO: multi-turn conversational/agent settings).
- Strong emphasis on evaluating *trajectory/protocol* rather than single-shot outputs.
- Highlights failure modes that only appear after repeated steps / long-horizon interaction.

## How is this different from GALILEO?

- Domain: software engineering agents + security vulnerabilities (CWE-focused), rather than conversational truthfulness/sycophancy/belief-drift style behaviors.
- Evaluation artifact: unit tests (functional + security) act as the judge, instead of dialog-based metrics / stance-change metrics.
- Primary failure mode is “passes functional tests but fails security tests”, rather than “agrees/drifts under pressure” (though conceptually analogous to constraint violations).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO’s protocol explicitly separates evidence-driven revision vs social-pressure drift, it may offer cleaner causal attribution than security-test pass/fail (which can conflate multiple engineering mistakes).
- GALILEO can likely report richer per-turn dynamics (e.g., when/why drift happens) rather than just endpoint unit test outcomes.

## Where GALILEO is weaker / needs to improve

- SusVibes provides a very concrete, automatically-checkable “hard constraint” evaluation (security tests) at repository scale; if GALILEO lacks similarly hard, automated validators, it may be easier to dispute results.
- SusVibes explicitly studies mitigation trade-offs (security vs functionality) via intervention prompts; GALILEO should similarly test whether interventions reduce the *primary capability*.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider framing some GALILEO evaluations as “constraint violations under outcome pressure” (analogy to secure-vs-functional trade-off), and add a metric for “passes task objective but violates constraint”.
- [ ] Add an ablation section mirroring SusVibes-style mitigation tests: interventions that attempt to reduce drift/sycophancy, and quantify the trade-off against base task success.
- [ ] In related work, cite SusVibes as evidence that multi-turn, agentic protocols reveal qualitatively different robustness/safety failures than single-turn benchmarks.

## Quotes / details to potentially cite

- SusVibes: “a benchmark consisting of 200 feature-request software engineering tasks from real-world open-source projects, which, when given to human programmers, led to vulnerable implementations.”
- “Although 61% of the solutions from SWE-Agent with Claude 4 Sonnet are functionally correct, only 10.5% are secure.”
- Key motivation: prior secure-code benchmarks are single-turn / small-context and do not reflect multi-turn coding agents operating on full repositories with environment feedback.
