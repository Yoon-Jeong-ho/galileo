# Workflow-R1: Group Sub-sequence Policy Optimization for Multi-turn Workflow Construction

- Year: 2026
- Venue: arXiv
- Authors: Zikun Qu; Zhongquan Zhou; Pengyu Liang; Xiang Li; Zhiwei Shang; Zhi Hong; Kaiyu Huang; Zhiyong Wang; Zhongxiang Dai
- URL: https://arxiv.org/abs/2602.01202
- BibTeX key (if we add it): WorkflowR1Qu2026
- Tags: workflows, agentic-reasoning, multi-turn, RL, GRPO, GSPO, optimization-granularity

## One-sentence takeaway

Workflow-R1 reframes workflow construction as a multi-turn natural-language sequential decision process and introduces GSsPO, an RL objective that optimizes over “Think-Action” sub-sequences (rather than tokens or whole trajectories) to better align learning with agent decision boundaries.

## What problem does it solve?

- Prior workflow-optimization methods often generate a full workflow/program “in one shot” (frequently as code) before seeing intermediate tool/step outcomes, yielding a rigid open-loop plan (“static execution trap”).
- Standard RL fine-tuning granularities are mismatched for multi-turn agent behavior:
  - Token-level (e.g., GRPO): can break semantic coherence across a think→act unit.
  - Whole-sequence (e.g., GSPO): credit assignment is too coarse for long-horizon multi-turn interactions.

## What is the core method / protocol?

- **Workflow-R1 framework:** treat workflow construction as a **closed-loop multi-turn interaction** in natural language with an interleaved cycle:
  - Think → Act (choose operator / step) → Observe (receive results) → repeat.
- **GSsPO (Group Sub-sequence Policy Optimization):**
  - Parse the model output/trajectory into **sub-sequences** corresponding to atomic decision units (in their setup, a Think-Action cycle).
  - Apply importance sampling / optimization **at the sub-sequence level** (geometric mean likelihood ratio over tokens in the sub-sequence), then average across sub-sequences.
  - Motivation: align gradient updates with semantic boundaries of decision units.
  - Paper claims an explicit design to mitigate “verbosity bias” by neutralizing confounds due to action cardinality/length.

## What are the key metrics?

- Task performance on multiple QA / reasoning benchmarks (reported as “seven benchmarks” in the intro/fig description), compared against:
  - vanilla prompting and
  - prior workflow optimization baselines.
- Learning robustness / efficiency for multi-turn reasoning under RL (qualitative claim).

## What are the main results?

- Workflow-R1 reportedly **outperforms competitive baselines** across multiple QA benchmarks and different backbone LLMs (intro mentions Qwen2.5-32B-Instruct and “DeepSeek V3.2”).
- Key empirical claim: optimizing at the Think-Action **sub-sequence** improves learning vs token-level and sequence-level alternatives in multi-turn settings.

## How is this similar to GALILEO?

- Both care about **agentic multi-step behavior** where the unit of progress is not a single token but a higher-level decision/action.
- Emphasizes **structure-aware optimization** that respects boundaries in an agent’s interaction protocol (e.g., reasoning step → action/tool call → observation).

## How is this different from GALILEO?

- This paper is primarily about **RL objective design / training** for workflow-construction policies, not (as far as reviewed here) a specific GALILEO-style algorithmic contribution around our particular setting.
- Workflow-R1 uses **natural-language workflow construction** (explicitly avoiding code-centric workflow programs) as a key design choice.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO already has a clean formalization of interaction units and evaluation beyond QA benchmarks, we may offer broader validation/clarity than their QA-centric presentation.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently trains with token-level RL or whole-trajectory objectives, we may be exposed to the same **granularity mismatch** / credit-assignment issues described here.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph contrasting **token-level (GRPO)** vs **sequence-level (GSPO)** vs **sub-sequence-level (GSsPO)** optimization for multi-turn agent protocols.
- [ ] Consider an ablation (even small-scale) where the optimization unit matches GALILEO’s atomic interaction cycle (our equivalent of Think-Act-Observe) and report impact on stability / success rate.
- [ ] Check whether GALILEO suffers from “verbosity bias” (longer action sequences getting favored) and whether per-unit normalization like GSsPO helps.

## Quotes / details to potentially cite

- “We reframe workflow optimization as a multi-turn conversation … interleaved cycle of Thinking, Acting, and Observing.” (Intro)
- “We propose Group Sub-sequence Policy Optimization (GSsPO) … align[s] the optimization granularity with the agent’s decision unit … the atomic Think-Action cycle.” (Abstract/§1)
- “Static Execution Trap” critique of one-shot workflow/program synthesis before observing intermediate outcomes. (Intro)
