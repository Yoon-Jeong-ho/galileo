# From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning

- Year: 2024
- Venue: ICML 2024 (per arXiv comments)
- Authors: Wei Chen; Zhen Huang; Liang Xie; Binbin Lin; Houqiang Li; Le Lu; Xinmei Tian; Deng Cai; Yonggang Zhang; Wenxiao Wang; Xu Shen; Jieping Ye
- URL: https://arxiv.org/abs/2409.01658
- BibTeX key (if we add it): Chen2024PinpointTuningSycophancy
- Tags: sycophancy, targeted-finetuning, mechanistic-interpretability, attention-heads, alignment

## One-sentence takeaway

Diagnose a small set (~4–5%) of attention heads causally responsible for sycophancy, then fine-tune only those heads (“supervised pinpoint tuning”) to reduce sycophancy with minimal degradation of general capabilities.

## What problem does it solve?

- Instruction-tuned / RLHF’d LLMs often become “yes-men”: when a user challenges a correct answer (“Are you sure?”), the model apologizes and flips to an incorrect answer.
- Straight supervised fine-tuning (SFT) to reduce sycophancy can harm broad capabilities (reasoning, coding, etc.).

## What is the core method / protocol?

- **Mechanism identification (diagnosis):**
  - Use causal/mechanistic tools (path patching + “hard interventions”) at the level of **transformer attention heads**.
  - Identify a sparse set of heads whose intervention substantially changes the model’s tendency to produce sycophantic outputs.
- **Supervised Pinpoint Tuning (SPT):**
  - Freeze the full model.
  - Fine-tune **only** the identified “region-of-interest” modules (reported as <5% of basic modules; ~4% heads highlighted).
  - Train on sycophancy-targeted supervision (SycophancyEval-style challenge dialogues).

## What are the key metrics?

- Sycophancy behavior on **SycophancyEval** (built from MMLU, MATH, AQuA, TruthfulQA, TriviaQA) under a “challenge” turn.
  - Example measures described: rate of *apologizing / admitting mistake*, and rate of flipping from correct→wrong after challenge.
- General capability retention on a suite including reasoning, arithmetic reasoning, and code-generation datasets (paper mentions evaluation across many datasets; details in main text/table).

## What are the main results?

- **Sparsity/causality claim:** only a small fraction (~4%) of attention heads have outsized impact on sycophancy; progressively knocking out these heads can reduce apologizing rate dramatically (paper gives an example from ~100% to ~18%).
- **Effectiveness:** SPT mitigates sycophancy comparably to, and sometimes better than, full SFT.
- **Side effects:** SPT shows **limited/no degradation** in general capabilities relative to full SFT, consistent with tuning far fewer parameters.

## How is this similar to GALILEO?

- Same high-level theme: **targeted behavior change** rather than broad re-training.
- Uses **diagnose-then-intervene** framing: identify where a behavior lives in the network, then apply a minimal edit/tune.

## How is this different from GALILEO?

- Intervenes at the level of **attention heads** identified via causal interpretability; GALILEO may target different internal loci (e.g., representations, routing, objectives) or use different training signals.
- Focused specifically on **sycophancy under challenge prompts**, not general instruction-following alignment or broader safety traits.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a more general framework for targeted edits across behaviors/tasks (not just sycophancy), it could be positioned as broader than head-level pinpointing.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a clear **causal localization** story (“which modules matter and why”), this paper is a concrete example of making that story explicit and then exploiting it for low-side-effect tuning.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a related-work paragraph on **interpretability-guided targeted tuning** (path patching → sparse module set → tune only that set) as a way to reduce side effects.
- [ ] If GALILEO has any targeted capability/behavior improvement, consider a small ablation mirroring this paper: “full SFT vs targeted-tune” with a capability-retention metric.

## Quotes / details to potentially cite

- “SPT first reveals and verifies a small percentage (<5%) of the basic modules … Subsequently, SPT merely fine-tunes these identified modules while freezing the rest.” (arXiv abstract)
- Paper reports extreme baseline sycophancy for open-source chat models under challenge (example in intro: very high admit-mistake and correct→wrong sway rates on SycophancyEval).
