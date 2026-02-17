# Training Long-Context, Multi-Turn Software Engineering Agents with Reinforcement Learning

- Year: 2025
- Venue: arXiv
- Authors: Alexander Golubev, Maria Trofimova, Sergei Polezhaev, Ibragim Badertdinov, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Sergey Abramov, Andrei Andriushchenko, Filipp Fisin, Sergei Skvortsov, Boris Yangel
- URL: https://arxiv.org/abs/2508.03501
- BibTeX key (if we add it): golubev2025training
- Tags: agents, software-engineering, long-context, multi-turn, reinforcement-learning, SWE-bench

## One-sentence takeaway

A practical two-stage recipe (execution-feedback RFT → synchronous DAPO RL) can significantly improve an *interactive*, long-context SWE agent, pushing Qwen2.5-72B from 11% to 39% Pass@1 on SWE-bench Verified.

## What problem does it solve?

- Most “RL for LLMs” work targets single-turn/bandit-like tasks (math, one-shot code), whereas real SWE agents are multi-step, stateful, and only sparsely rewarded.
- They aim to show RL can work in this non-degenerate setting: long-horizon interaction with rich observations (logs, tests) and terminal success/failure.

## What is the core method / protocol?

- Formulate SWE agenting as a POMDP where actions are tool/command strings; observations are execution outputs; reward is sparse terminal success (tests pass).
- Two-stage training:
  - **Rejection Fine-Tuning (RFT)** using execution feedback to train instruction/format/tool-use competence.
  - **Synchronous RL with DAPO** (a GRPO-style, critic-free policy optimization variant) for iterative improvement.
- DAPO stabilizers (as described): asymmetric clipping (“clip higher”), dynamic sampling (drop zero-advantage samples), soft overlong punishment, token-level loss averaging.

## What are the key metrics?

- Pass@1 on **SWE-bench Verified**.
- Pass@1 on **SWE-rebench** (May/June splits).

## What are the main results?

- Base model: **Qwen2.5-72B-Instruct**.
- SWE-bench Verified Pass@1:
  - 11% (base) → 20% (RFT) → **39% (RFT + DAPO RL)**.
- SWE-rebench Pass@1:
  - **35% (May)**, **31% (June)**.
- Claim: competitive with larger open(-ish) models (e.g., DeepSeek-V3-0324, Qwen3-235B-A22B) on those splits.

## How is this similar to GALILEO?

- Focus on **multi-turn, stateful environments** where observations matter and outcomes are verifiable.
- Emphasizes **long-context** and long-horizon coherence across many steps.
- Uses *protocol-level* training/eval framing (POMDP, terminal reward), which is often relevant when defining robust evaluation pipelines.

## How is this different from GALILEO?

- Domain is **software engineering agenting** (tool-using code repair), not belief/behavioral stability per se.
- Optimization target is **task success (tests pass)**, not stability/faithfulness under conversational pressure or drift metrics.
- Their “multi-turn” difficulty is largely environment interaction + sparse reward, rather than adversarial/social pressure dynamics.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is about *stability/behavioral drift*, this paper provides little in terms of *measurement* for drift vs evidence-driven revision; it mainly optimizes reward.

## Where GALILEO is weaker / needs to improve

- If GALILEO ever needs to argue “RL can shape multi-turn agent behaviors,” this is a strong contemporary exemplar in a real interactive setting.
- Might motivate including RL-based baselines or at least discussing why we avoid reward-hacking / test-overfitting analogues.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related-work paragraph: contrast “bandit-style RL for LLMs” vs *interactive POMDP* RL; cite this as evidence RL scales to long-context, multi-turn tool-use.
- [ ] Consider whether any DAPO stabilizers (dynamic sampling / length penalties / asymmetric clipping) have analogues for training stability-focused conversational agents.

## Quotes / details to potentially cite

- “Applying this pipeline to Qwen2.5-72B-Instruct, we increase its Pass@1 on the SWE-bench Verified benchmark from 11% to 39%…”
- “Our methodology begins with rejection fine-tuning (RFT) using execution feedback… followed by a synchronous RL pipeline using DAPO…”
- Motivation framing: single-turn tasks are “degenerate” multi-turn MDPs with no intermediate environment feedback; SWE requires stateful interaction.
