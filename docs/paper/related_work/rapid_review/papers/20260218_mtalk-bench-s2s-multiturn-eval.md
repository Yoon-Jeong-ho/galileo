# MTalk-Bench: Evaluating Speech-to-Speech Models in Multi-Turn Dialogues via Arena-style and Rubrics Protocols

- Year: 2025
- Venue: arXiv
- Authors: Yuhao Du, Qianwei Huang, Guo Zhu, Zhanchen Dai, Shunian Chen, Qiming Zhu, Le Pan, Minghao Chen, Yuhao Zhang, Li Zhou, Benyou Wang, Haizhou Li
- URL: https://arxiv.org/abs/2508.18240
- BibTeX key (if we add it): mtalkBenchDu2025
- Tags: speech-to-speech, multi-turn, evaluation, benchmark

## One-sentence takeaway

MTalk-Bench proposes a multi-turn **speech-to-speech** benchmark plus a dual evaluation protocol (pairwise arena + rubric scoring), finding S2S models handle semantics well but struggle with paralinguistics/ambient sounds and that judge reliability is fragile to **length/position** biases.

## What problem does it solve?

- Existing evaluation for speech-to-speech (S2S) LLMs is weak for **complex multi-turn dialogues**, especially beyond “semantic correctness”.
- Need scenarios and metrics that capture: (i) semantic content, (ii) paralinguistic signals, and (iii) ambient sound understanding, and that work for both **relative** and **absolute** scoring.

## What is the core method / protocol?

- **MTalk-Bench**: multi-turn S2S benchmark organized into 3 dimensions:
  - Semantic Information
  - Paralinguistic Information
  - Ambient Sound
- Each dimension includes **9 realistic scenarios**, with “targeted tasks” to probe specific capabilities (including reasoning).
- **Dual evaluation**:
  - **Arena-style**: pairwise comparisons (relative ranking)
  - **Rubrics-based**: absolute scoring against criteria
- Collects **model and human outputs**; uses both **human evaluators** and **LLM-as-a-judge**.

## What are the key metrics?

- Arena-style pairwise win/loss (relative preference).
- Rubric scores (absolute), across the three information dimensions.
- Judge reliability analysis (agreement / when rankings become distinguishable) and bias diagnostics (length/position).

## What are the main results?

- S2S models:
  - Strong on **semantic** processing.
  - Weak on **paralinguistic** information and **ambient** sound perception.
  - Tend to “regain coherence” by **increasing response length**, trading off efficiency in multi-turn settings.
  - “Modality-aware, task-specific designs” outperform brute scaling.
- Evaluation reliability:
  - Arena and Rubrics produce **consistent but complementary** rankings; clear distinctions only when performance gaps are large.
  - LLM judges align with humans when criteria are explicit or gaps are large, but show **position and length biases**.
  - For nonverbal evaluation, LLM-as-judge is only reliable when provided **text annotations**.

## How is this similar to GALILEO?

- Shares the theme of **multi-turn evaluation protocols** and highlighting that naïve aggregate metrics can hide important failure modes.
- Emphasizes evaluator/metric **reliability pitfalls** (judge biases), which is directly relevant to how GALILEO should justify its evaluation methodology.

## How is this different from GALILEO?

- Focuses on **speech-to-speech** interaction and perceptual channels (paralinguistics, ambient sound), not primarily on belief drift / social pressure / evidence vs persuasion controls.
- Uses arena + rubric scoring rather than explicitly time-to-event / survival-style multi-turn stability metrics (at least in the abstract-level description).

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO targets belief drift / persuasion robustness: we can offer cleaner **causal controls** (pressure vs evidence) and trajectory-level outcomes (flip/recovery) that are orthogonal to audio perception.

## Where GALILEO is weaker / needs to improve

- If our evaluation uses LLM judges, we need an explicit section on **judge bias** (length/position) and when rankings are statistically meaningful—MTalk-Bench provides language and empirical claims we can cite.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short “judge reliability” subsection: when pairwise vs rubric-style evaluation agree, and how we mitigate **length/position** bias.
- [ ] Consider reporting both a **relative** (pairwise) and **absolute** (rubric) score for key multi-turn outcomes, or justify why one suffices.
- [ ] If we ever extend to spoken interaction, reuse their dimension split (semantic vs paralinguistic vs ambient).

## Quotes / details to potentially cite

- “MTalk-Bench, a multi-turn S2S benchmark covering three core dimensions: Semantic Information, Paralinguistic Information, and Ambient Sound.”
- “Arena-style evaluation (pairwise comparison) and Rubrics-based evaluation (absolute scoring) for relative and absolute assessment.”
- “LLM-as-a-judge aligns with humans when gaps are clear or criteria explicit, but exhibits position and length biases.”
