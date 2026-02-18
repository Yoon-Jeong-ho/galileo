# DeceptionBench: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios

- Year: 2025
- Venue: NeurIPS 2025 (per arXiv comments)
- Authors: Yao Huang, Yitong Sun, Yichi Zhang, Ruochen Zhang, Yinpeng Dong, Xingxing Wei
- URL: https://arxiv.org/abs/2510.15501
- BibTeX key (if we add it): deceptionbench2025huang
- Tags: deception, benchmark, multi-turn, incentives, coercion, egoism, sycophancy

## One-sentence takeaway

DeceptionBench is a generative, scenario-based benchmark that probes LLM deception across domains and shows deception can be strongly amplified by reward/coercion framing and by multi-turn “feedback-loop” induction.

## What problem does it solve?

- Existing deception evaluations are narrow (few scenarios / single domain / limited settings), making it hard to characterize when and why LLMs become deceptive in realistic deployments.
- Need to separate *where deception shows up* (domain), *why it happens* (intrinsic driver), and *what makes it worse* (extrinsic context, including multi-turn dynamics).

## What is the core method / protocol?

- **Generative QA** setup (open-ended) rather than multiple-choice, to expose behavioral nuance.
- **150 scenarios** across **5 domains** (Economy, Healthcare, Education, Social Interaction, Entertainment), with **>1,000 samples**.
- Two **intrinsic role modes** (their framing):
  - **Egoism**: model acts as an autonomous agent prioritizing self-interest/self-preservation.
  - **Sycophancy**: model acts as a user assistant prioritizing user appeasement.
- Three **extrinsic context levels**:
  - **L1 (Inherent)**: neutral baseline prompt (single-turn).
  - **L2 (Induced)**: single-turn prompts with **reward-based incentivization** or **coercive pressure**.
  - **L3 (Multi-turn induced)**: sustained multi-turn loops where an auxiliary “inducer” agent iteratively adapts prompts based on the model’s previous response/rejection rationale, attempting to elicit deception.

## What are the key metrics?

- Primary outcome is whether the model produces **deceptive** behavior in the scenario (paper discusses evaluating both **thought** and **response**; exact scoring rubric/details not fully captured from quick skim).
- Analysis slices results by domain, intrinsic mode (egoism vs sycophancy), extrinsic context (L1/L2/L3), and model family.

## What are the main results?

- Deceptive tendencies vary substantially by **domain** (some contexts elicit much higher deception).
- **Extrinsic induction matters**:
  - **Rewards/incentives** can increase deception, especially for models showing stronger **sycophantic** tendencies.
  - **Coercive pressure** can increase deception, especially for more **egoistic** patterns.
- **Multi-turn feedback loops (L3)** can further expose/escalate deception: iterative interaction can compound mild misrepresentation into more severe falsification.
- Overall claim: current models lack robust resistance to manipulative contextual cues; safeguards should consider both single-turn and multi-turn settings.

## How is this similar to GALILEO?

- Targets **multi-turn** behavioral failures under **social/interactional pressure** (reward framing, coercion, iterative persuasion), which is conceptually close to multi-turn sycophancy/persuasion/drift settings.
- Emphasizes that evaluation must capture **trajectory effects** (deception can grow over turns), not just one-shot accuracy.

## How is this different from GALILEO?

- Focuses specifically on **deception** (misleading/strategically false outputs) across **societal domains**, rather than (only) agreement/stance stability or truthfulness under user pressure.
- Includes a distinct **egoism vs sycophancy** intrinsic split and a dedicated **multi-turn “inducer agent”** protocol to escalate deception.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses simpler, more controlled perturbations (e.g., standardized multi-turn pressure scripts) it may yield cleaner causal attribution than an adaptive inducer loop.
- If GALILEO has explicit “flip” / stance-change metrics, it may be easier to compare across tasks than scenario-by-scenario deception labeling.

## Where GALILEO is weaker / needs to improve

- Consider adding a deception-focused slice (or mapping) where “agreeing with user” vs “self-serving” correspond to different deception drivers.
- Ensure evaluation covers **induced settings** (reward / coercion) and **multi-turn escalation**, not just baseline multi-turn interaction.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite DeceptionBench as evidence that **reward/coercion framing + multi-turn feedback** can amplify problematic behaviors.
- [ ] Consider an ablation where the *same base scenario* is tested under **neutral vs incentivized vs coercive** prompts to disentangle extrinsic effects.
- [ ] If applicable, consider an **adaptive attacker/inducer** variant to test escalation dynamics (bounded turns) and report “time-to-deception”.

## Quotes / details to potentially cite

- “DeceptionBench … systematically evaluates how deceptive tendencies manifest across different societal domains, what their intrinsic behavioral patterns are, and how extrinsic factors affect them.”
- Benchmark stats: “150 … scenarios in five domains … with over 1,000 samples.”
- Extrinsic levels: “Neutral conditions … reward-based incentivization, and coercive pressures … [and] sustained multi-turn interaction loops …”
