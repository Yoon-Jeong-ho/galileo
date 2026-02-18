# From Assistant to Double Agent: Formalizing and Benchmarking Attacks on OpenClaw for Personalized Local AI Agent

- Year: 2026
- Venue: arXiv
- Authors: Yuhang Wang; Feiming Xu; Zheng Lin; Guangyu He; Yuzhe Huang; Haichang Gao; Zhenxing Niu; Shiguo Lian; Zhaoxiang Liu
- URL: https://arxiv.org/abs/2602.08412
- BibTeX key (if we add it): wang2026pasb
- Tags: agent-security, benchmark, prompt-injection, indirect-prompt-injection, tool-attacks, memory-poisoning, long-horizon

## One-sentence takeaway

PASB is an end-to-end, black-box benchmark for *personalized* tool-using agents (scenarios + canary assets + long-horizon traces) and shows substantial residual attack success on OpenClaw even under common “prompt-layer” defenses.

## What problem does it solve?

- Existing agent-security benchmarks often focus on synthetic/task-centric setups and miss the *real* attack surface of personalized assistants: persistent state, long-horizon interactions, private assets, and high-privilege tools.
- Need evaluation that measures *system-level harms* (leakage, unsafe actions, persistence) rather than only “bad text outputs”.

## What is the core method / protocol?

- Define a personalized agent as a persistent policy over mixed-trust observations (user input + external content + tool outputs + retrieved memory).
- Define attack tasks \(\Gamma\) with: scenario/context, injection channels, interaction budget, adversarial goal class, and an observable-trace success predicate.
- Observable trace includes: user inputs, textual responses, tool/skill calls + arguments, and tool returns.
- Success predicate is an OR over:
  - **Leakage**: canary/private assets appear in response/tool args/tool returns.
  - **Unsafe action**: a tool-call violates scenario policy \(\mathcal{F}\).
  - **Persistence**: harm occurs after attacker stops injecting (captures long-horizon propagation).
- Scenario suite (personalized realism):
  - **A External content hub** (agent reads untrusted web/community content).
  - **B Personal context + long-term memory management** (poisoning/extracting/modifying memory).
  - **C Skills/plugins and tool-return risks** (over-trusting tool outputs; deceptive tool returns).
- Attack primitives instantiated end-to-end:
  - Direct prompt injection, indirect injection via external content, tool-return deception, memory poisoning.
- Case study: run OpenClaw in deployed form with a harness that drives trajectories and validates success via *actual runtime/environment effects* (not just parsing text).

## What are the key metrics?

- **ASR (Attack Success Rate)**: task-specific harm success (e.g., *target* skill/tool call occurs).
- **Resp Rate**: whether the agent triggers *any* skill/tool call (used in IPI simulation).
- Memory tasks (Scenario B):
  - **STM-Extract / LTM-Extract success rate**.
  - **STM-Edit / LTM-Edit Write Success Rate (WSR)**, verified via markers in the agent’s file system.

## What are the main results?

(From the paper’s reported tables/summary; emphasis on what’s citable for our related work.)

- In IPI-style attacks, **Resp Rate remains high** across settings (reported range **93.8%–99.0%**), suggesting attacks tend to *steer which tool is called* rather than preventing tool use.
- Under **no defense**, a “Combined Attack” achieves high ASR across backbones:
  - **66.8%** (Llama-3.1-70B-Instruct)
  - **52.7%** (Qwen2.5-7B-Instruct)
  - **61.9%** (gpt-4o-mini)
- Prompt-layer defenses reduce but do not eliminate ASR:
  - For Llama-3.1-70B under Combined Attack: **66.8% → 33.5% (Delimiter) → 22.0% (Sandwich)**.
  - Even with Sandwich, reported residual ASR remains **10.5%–22.0%** across models/attacks.
- Memory risks: **LTM extraction success > STM extraction success** (qualitative claim; tables referenced), and defenses reduce extraction/modification but leave non-zero residual success.
- Implementation detail worth citing: they validate attacks via **actual executed tool/skill code in an isolated runtime**, counting success only if it produces tangible environment effects.

## How is this similar to GALILEO?

- Both care about *realistic* tool-using agent behavior (not just static LM outputs) and evaluation that incorporates tool calls and state.
- PASB’s emphasis on **mixed-trust inputs**, **high-privilege toolchains**, and **persistent memory** aligns with the risk model GALILEO must assume for real deployments.

## How is this different from GALILEO?

- PASB is primarily a **security evaluation benchmark** (attack tasks + ASR/WSR), not an agent architecture/method for robust execution.
- It focuses on **prompt-injection / tool-return deception / memory poisoning** and defense baselines (delimiter/sandwich/instruction prevention), rather than GALILEO’s core algorithmic contributions (whatever GALILEO’s method is in the paper).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides *principled* controls on tool execution (e.g., capability restrictions, typed action policies, guardrails that enforce policies at runtime), that would address PASB’s main finding that prompt-layer defenses leave residual ASR.
- If GALILEO includes systematic evaluation across tasks beyond security, it can position PASB as complementary (security-specific).

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks an end-to-end *black-box* security evaluation protocol with auditable canaries and persistence metrics, PASB is a strong “related work / missing eval” pointer.
- If GALILEO does not explicitly test **memory write/read attacks** and **tool-return deception**, PASB suggests these are critical to include.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite PASB as evidence that delimiter/sandwich defenses are insufficient (residual ASR ~10–22%).
- [ ] Consider adding a small PASB-inspired evaluation slice: (a) indirect injection via retrieved web content; (b) tool-return payload; (c) memory extraction/modification with canary markers; measure persistence.
- [ ] If GALILEO has runtime policy enforcement, contrast with PASB’s prompt-layer baselines and argue why GALILEO should reduce ASR/WSR further.

## Quotes / details to potentially cite

- PASB models success over an observable trace and defines success as leakage OR unsafe tool action OR persistence after attacker stops injecting.
- IPI metrics: Resp Rate (any tool call) vs ASR (target skill call).
- Reported IPI results summary: Resp Rate 93.8%–99.0%; Combined Attack ASR up to 66.8% (no defense); Sandwich reduces but leaves 10.5%–22.0% residual ASR.
- Memory tasks: STM/LTM extraction and STM/LTM edit WSR verified by markers in the agent file system.
