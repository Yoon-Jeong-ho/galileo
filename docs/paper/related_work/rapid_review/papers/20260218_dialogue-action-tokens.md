# Dialogue Action Tokens: Steering Language Models in Goal-Directed Dialogue with a Multi-Turn Planner

- Year: 2024
- Venue: arXiv
- Authors: Kenneth Li; Yiming Wang; Fernanda Viégas; Martin Wattenberg
- URL: https://arxiv.org/abs/2406.11978
- BibTeX key (if we add it): li2024dat
- Tags: multi-turn, planning, steering, offline-RL, prefix-tuning, red-teaming, social-simulation

## One-sentence takeaway

Freeze the base LM and train a small RL “planner” that emits a tiny continuous action vector mapped into a couple of learned prefix tokens each turn, yielding large gains on goal-directed multi-turn dialogue (Sotopia) while also enabling stronger multi-turn red-teaming attacks.

## What problem does it solve?

- How to adapt an LM agent to pursue *long-horizon, goal-directed* objectives in multi-turn dialogue without full RLHF-style finetuning that can cause “language degradation” under reward over-optimization.
- How to represent dialogue planning in a tractable action/state space for standard RL algorithms.

## What is the core method / protocol?

- Formulate 2-party dialogue as an MDP:
  - State: scenario description + dialogue history.
  - Action: an utterance.
  - Reward: provided by a judge model (task-dependent).
- DAT (Dialogue Action Tokens):
  - Keep the pretrained LM frozen.
  - Extract a continuous state vector from the LM (the last-layer, last-token embedding of the dialogue history).
  - Train a small planner network (MLP) that outputs a low-dim continuous action vector (d′=64).
  - Map that action vector through an “up-mapping” matrix W into L prefix token embeddings (L=2) that are prepended to the prompt for controlled generation each turn.
- Two-stage training pipeline:
  1) **Self-cloning**: train planner + W from scratch to reproduce the *unsteered* LM’s behavior (i.e., get a good initialization that preserves language quality).
  2) **Offline RL**: freeze W and finetune the planner with TD3+BC on an offline buffer of episodes to maximize the judge reward.

## What are the key metrics?

- **Sotopia social simulation**: GPT-4-judged aggregated score over dimensions (goal completion, believability, knowledge, secret keeping, relationship maintenance, social rule obedience, material benefits).
- **Multi-turn red teaming (HarmBench-derived)**:
  - Average success rate (ASR, %) of eliciting harmful answers after multi-turn attacker–defender dialogue.
  - Judge is a fine-tuned Llama-2-13B HarmBench evaluator; reward softened via logit diff of “Yes” vs “No”.

## What are the main results?

- **Sotopia (3-round dialogues; controlled agent is Llama-2-7B-chat)**:
  - Unsteered baseline vs DAT (partner = Llama-2-7B-chat):
    - Baseline: 3.24 ± 0.14
    - DAT: **3.59 ± 0.13**
  - With stronger partner (partner = Llama-3-8B-instruct):
    - DAT: **3.73 ± 0.14**
  - Reported as surpassing GPT-4 in this setting:
    - GPT-4 row shows 3.53 ± 0.14 (partner Llama-2-7B-chat) and 3.65 ± 0.12 (partner Llama-3-8B-instruct).
- **Red teaming (attacker = Llama-3-8B-instruct steered with DAT; 3-step dialogues)**:
  - Against Llama-3-8B-instruct defender (training-time defender):
    - Multi-turn baseline attacker: 5.03 ± 1.73
    - DAT attacker: **28.93 ± 3.60**
  - Against Llama-2-7B-chat defender (generalization test):
    - DAT attacker: **18.87 ± 3.11**
  - For comparison, single-turn GCG is ~18.87 ± 3.11 vs Llama-3 defender and ~32.08 ± 3.70 vs Llama-2-7B defender (paper table).

## How is this similar to GALILEO?

- Both are centrally about **multi-turn interaction dynamics** (long-horizon behavior, compounding effects across turns).
- The red-teaming setup highlights **multi-turn attack surfaces** and how attackers can strategically steer outcomes over rounds—adjacent to GALILEO’s concerns about pressure/persuasion/drift across turns.
- Reinforces the idea that “trajectory-level” evaluation (not single-turn) can reveal vulnerabilities that are invisible in one-shot tests.

## How is this different from GALILEO?

- DAT is primarily a **capability/steering method** (train a planner to optimize a reward), whereas GALILEO is focused on **robustness under pressure** (sycophancy, persuasion, belief revision vs drift controls) and measuring/mitigating unwanted shifts.
- DAT assumes access to a reward/judge for the target objective; GALILEO’s target is often *truthfulness/independence under social pressure*, which is harder to specify as a stable scalar reward without Goodhart issues.
- Their “state” is an internal embedding summary; GALILEO may emphasize explicit constraints/controls/diagnostics for stability and principled belief revision.

## Where GALILEO is stronger / cleaner (if true)

- Likely stronger at **explicitly characterizing undesirable multi-turn failure modes** (drift, flip-flops, sycophancy) rather than optimizing for a downstream goal that could inadvertently induce them.
- The DAT paper’s heavy reliance on **LLM-as-judge** reward signals (GPT-3.5/4, HarmBench judge) creates potential confounds; GALILEO can position itself as more diagnostic/measurement-centric.

## Where GALILEO is weaker / needs to improve

- If GALILEO is purely evaluative, DAT is a concrete, lightweight **intervention mechanism**: “small module steers frozen LM each turn,” which could inspire GALILEO-style *controls* that do not require full finetuning.
- DAT suggests an avenue to build **turn-by-turn control policies** (planner) that could enforce stability/anti-sycophancy objectives over a trajectory.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider citing DAT as evidence that (a) multi-turn dialogue can be framed as an MDP and (b) small learned controllers can strongly change multi-turn behavior without updating the base LM.
- [ ] Add a short discussion: multi-turn *attackers* can be made much stronger with planning/steering (DAT red-teaming results), motivating GALILEO’s focus on robustness across rounds.
- [ ] Method idea: adapt the “planner + tiny prefix embeddings per turn” concept to enforce **anti-sycophancy / independence** objectives (e.g., reward for maintaining calibrated disagreement when user is wrong, reward for evidence-consistent belief revision), while monitoring for language degradation / Goodharting.
- [ ] Evaluation idea: compare robustness of “frozen LM + stability-planner” vs base LM under the same pressure/persuasion multi-turn protocols.

## Quotes / details to potentially cite

- “We sidestep [language degradation] by training a small planner model … All LM parameters are kept frozen.”
- “We propose Dialogue Action Tokens (DAT) … treat each utterance as an action … converting dialogues into games where … reinforcement learning can be applied.”
- Sotopia result claim: “DAT-steered LLaMA model surpasses GPT-4’s performance.”
- Safety angle: applying DAT to an attacker LM in multi-turn red teaming “revealing a potential new attack surface.”
