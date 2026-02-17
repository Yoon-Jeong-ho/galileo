# PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts

- Year: 2023 (latest arXiv v5: 2024-07)
- Venue: arXiv (technical report)
- Authors: Kaijie Zhu; Jindong Wang; Jiaheng Zhou; Zichen Wang; Hao Chen; Yidong Wang; Linyi Yang; Wei Ye; Yue Zhang; Neil Zhenqiang Gong; Xing Xie
- URL: https://arxiv.org/abs/2306.04528
- BibTeX key (if we add it): zhu2023promptrobust
- Tags: robustness, adversarial-prompts, prompt-perturbation, evaluation

## One-sentence takeaway

Small, semantics-preserving perturbations to *prompts* (typos/synonyms/paraphrases) can substantially degrade LLM performance across many tasks, motivating prompt-level robustness evaluation beyond sample-level adversarial testing.

## What problem does it solve?

- Prior robustness work largely targets *samples* (adversarial examples / OOD inputs), but in practice a single prompt is reused across many samples; prompt perturbations (accidental or adversarial) can therefore induce broad failure.
- Need a systematic benchmark to measure LLM sensitivity to realistic prompt perturbations at multiple granularities.

## What is the core method / protocol?

- Build **PromptRobust**, a benchmark suite that:
  - Enumerates prompt styles (described as: zero-shot, few-shot, role-oriented, task-oriented).
  - Applies **prompt attacks** at multiple levels:
    - Character-level (e.g., typos)
    - Word-level (e.g., synonym substitutions)
    - Sentence-level (paraphrases)
    - Semantic-level (meaning-preserving re-expressions; plus more adversarially crafted variants)
  - Evaluates multiple LLMs (ranging from instruction-tuned smaller models to ChatGPT/GPT-4) on a collection of standard NLP + reasoning tasks.
- Report robustness and analyze factors/transferability; provide practical recommendations for writing more robust prompts.

## What are the key metrics?

- Task performance under **clean prompts** vs **perturbed/adversarial prompts** (e.g., accuracy/F1/EM depending on the task).
- Robustness is essentially the *degradation gap* (or relative drop) induced by prompt perturbations; the paper also discusses transferability patterns of prompt attacks.

## What are the main results?

- Contemporary LLMs show **non-trivial performance drops** under slight prompt perturbations (even those mimicking natural user errors).
- Prompt perturbations can be more impactful than single-sample adversarial inputs because prompts are reused across many queries.
- Robustness varies by:
  - attack type (char/word/sentence/semantic)
  - prompt style (ZS/FS/role/task)
  - model family/scale
  - task

## How is this similar to GALILEO?

- Same high-level framing: robustness in *interactive / instruction-following* settings depends on stability under small changes to the conversational “control surface” (prompt/context).
- Emphasizes evaluation protocols that stress models with **semantics-preserving perturbations** rather than only distribution shift in content.

## How is this different from GALILEO?

- PromptRobust focuses on **single-shot prompt strings** (perturbed once) evaluated across tasks/datasets; GALILEO is oriented toward **multi-turn interaction dynamics** (drift, pressure, recovery, trajectory-level failure).
- Attacks are primarily textual perturbations of prompt wording, not sequential conversational manipulations.

## Where GALILEO is stronger / cleaner (if true)

- Can capture **time-to-failure**, recovery, and compounding effects over turns—failure modes that prompt-only perturbation benchmarks miss.
- More aligned to “agentic” / long-horizon use where context accumulates.

## Where GALILEO is weaker / needs to improve

- Could under-emphasize **prompt-surface robustness** (typos/synonyms/paraphrases) that affects real deployments and can be tested cheaply at scale.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “prompt-surface perturbation” slice: apply char/word/sentence perturbations to *system/developer* instructions and measure drift/failure over multi-turn trajectories.
- [ ] Consider reporting a robustness summary like: clean score vs perturbed score, plus turn-level degradation curves.
- [ ] In related work, position PromptRobust as the prompt-level analogue of adversarial example benchmarks.

## Quotes / details to potentially cite

- Prompt robustness is under-explored despite prompts being reused across many samples; therefore a perturbed prompt can have larger impact than an adversarial sample.
- PromptRobust uses prompt attacks across **character/word/sentence/semantic** levels and evaluates across multiple tasks (sentiment, NLI, RC, MT, math, etc.).
