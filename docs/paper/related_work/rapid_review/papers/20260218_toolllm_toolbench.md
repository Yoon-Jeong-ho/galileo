# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs

- Year: 2023
- Venue: arXiv
- Authors: Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun
- URL: https://arxiv.org/abs/2307.16789
- BibTeX key (if we add it): toollm-qin-2023
- Tags: tool-use, api, instruction-following, evaluation, dataset

## One-sentence takeaway

ToolLLM introduces an end-to-end framework (ToolBench + ToolEval + fine-tuned ToolLLaMA + API retrieval) to train/evaluate open LLMs for real-world REST API tool use at scale (~16k APIs).

## What problem does it solve?

- Open-source LLMs lag behind closed models (e.g., ChatGPT) in *tool-use* (choosing/calling external APIs to satisfy instructions).
- Existing instruction tuning under-emphasizes tool-use, and evaluation is hard because correctness depends on multi-step API-call trajectories.

## What is the core method / protocol?

- ToolLLM framework components:
  - **ToolBench**: instruction-tuning dataset for tool use, constructed automatically with ChatGPT.
    - Stage (i) **API collection**: 16,464 real-world RESTful APIs across 49 categories (from RapidAPI Hub).
    - Stage (ii) **instruction generation**: ChatGPT generates diverse single-tool and multi-tool instructions grounded in collected APIs.
    - Stage (iii) **solution path annotation**: ChatGPT is used to produce a feasible solution path (sequence/chain of API calls) per instruction.
  - **Decision tree search for reasoning traces**: a depth-first-search style decision-tree algorithm to expand the space of reasoning traces and select better ones (intended to improve tool-use reasoning robustness).
  - **ToolLLaMA**: fine-tune LLaMA on ToolBench for tool-use.
  - **Neural API retriever**: recommend candidate APIs for each instruction (supporting tool selection among many possible APIs).
  - **ToolEval**: automatic evaluator for tool-use (to score model tool-use performance without fully manual judging).

## What are the key metrics?

- Automatic evaluation via **ToolEval** on ToolBench tasks (tool selection + multi-step tool-call success).
- Generalization to unseen APIs and out-of-distribution evaluation on **APIBench** (as described in the abstract).

## What are the main results?

- ToolLLaMA (LLaMA fine-tuned on ToolBench + API retriever) shows strong performance on complex instructions and generalizes to unseen APIs.
- Reported as **comparable to ChatGPT** on their tool-use evaluations, and strong zero-shot generalization on OOD tool-use data (APIBench).

## How is this similar to GALILEO?

- Similar *evaluation framing*: building benchmarks/evaluators aimed at measuring a model capability (here: tool-use) with structured protocols.
- Similar *multi-step behavior* emphasis: tool-use requires multi-turn / multi-action trajectories; GALILEO likely cares about stability/robustness over sequences.

## How is this different from GALILEO?

- Focuses on **tool-use capability** (API calling, retrieval, execution paths), not directly on belief stability / truth maintenance / drift dynamics.
- The key objects are *APIs and call traces*, rather than *claims, evidence, and belief updates*.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets epistemic stability/robustness, it can provide clearer notions of “correctness” tied to truth/evidence rather than tool-call success.
- GALILEO can emphasize evaluator validity against extraction artifacts, whereas tool-use evaluation often conflates reasoning with tool availability/retrieval.

## Where GALILEO is weaker / needs to improve

- ToolLLM highlights that scaling to many “actions” needs infrastructure (retrieval over action space, trajectory evaluation). If GALILEO needs to scale interventions/actions, similar infra may be necessary.

## Action items for GALILEO (experiments / method / writing)

- [ ] In related work, cite ToolLLM/ToolBench as an example of: (a) large-scale synthetic data generation with strong teacher model, and (b) automatic evaluation for multi-step agentic behaviors.
- [ ] If GALILEO includes any action space (e.g., interventions/prompts/tools), consider whether an explicit retrieval module and trajectory evaluator would improve scalability and reproducibility.

## Quotes / details to potentially cite

- “We collect 16,464 real-world RESTful APIs spanning 49 categories from RapidAPI Hub … [and] prompt ChatGPT to generate diverse instructions … [and] search for a valid solution path (chain of API calls) for each instruction.” (abstract)
- “To evaluate the tool-use capabilities … we develop an automatic evaluator: ToolEval.” (abstract)
