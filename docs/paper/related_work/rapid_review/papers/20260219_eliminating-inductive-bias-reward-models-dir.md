# Eliminating Inductive Bias in Reward Models with Information-Theoretic Guidance

- Year: 2025
- Venue: arXiv
- Authors: Zhuo Li; Zhechao Yu; Feifei Tong; Anningzhe Gao; Tsung-Hui Chang; Xiang Wan; Erchao Zhao; Xiaoxi Jiang; Guanjun Jiang
- URL: https://arxiv.org/abs/2512.23461
- BibTeX key (if we add it): li2025dir
- Tags: reward-model, inductive-bias, debiasing, mutual-information, information-bottleneck, rlhf

## One-sentence takeaway

DIR debiases reward models by explicitly maximizing mutual information with preference labels while minimizing mutual information with known bias attributes (e.g., length/sycophancy/format), improving RLHF generalization.

## What problem does it solve?

- Reward models trained on human preference data pick up spurious “inductive biases” (length preference, stylistic/format cues, sycophancy), enabling reward hacking and poor generalization.
- Prior debiasing methods are either (a) bias-specific, or (b) rely on simple linear correlation measures (e.g., Pearson), missing non-linear / richer bias relationships.

## What is the core method / protocol?

- Formulate debiasing as an information-theoretic objective:
  - Maximize MI between RM scores (or preference predictions from RM scores) and human preference labels for response pairs.
  - Minimize MI between RM outputs and explicit bias attributes of the preference inputs.
- Use variational MI bounds to make it tractable:
  - Barber–Agakov (BA) lower bound for MI maximization.
  - CLUB upper bound for MI minimization.
- Add a “comparative” regularizer operating on *relative* bias attributes between the chosen/rejected responses in a pair (instead of per-response absolute bias features), to better match preference-pair supervision.

## What are the key metrics?

- Bias mitigation under controlled bias settings:
  - Response length bias
  - Sycophancy bias
  - Format / stylistic bias
- Downstream alignment and RM quality on common benchmarks mentioned in the paper:
  - Capability / preference-relevant evals: GSM8K, MMLU, ArenaHard, MT-Bench
  - RM benchmarks: RM-Bench, RewardBench

## What are the main results?

- DIR reduces targeted bias signals while also improving RLHF performance and generalization on multiple benchmarks (as reported by the authors).
- Claimed robustness benefit vs correlation-based regularizers and vs generic compression-only approaches (e.g., IB-style) because DIR explicitly penalizes information about bias attributes.

## How is this similar to GALILEO?

- Same high-level goal: produce more reliable reward signals by preventing the RM from exploiting spurious correlations in preference data.
- Emphasizes generalization / robustness across tasks rather than optimizing for a single benchmark.

## How is this different from GALILEO?

- DIR assumes bias attributes are identifiable/constructible (length, format descriptors, sycophancy signals) and then explicitly minimizes MI with those attributes.
- The objective is MI-based with variational bounds (BA/CLUB), which is a specific optimization toolkit rather than (e.g.) architectural changes, data interventions, or alternative preference objectives.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO can mitigate biases *without* explicitly enumerating bias attributes, it may be more generally applicable when biases are unknown or hard to quantify.
- If GALILEO avoids auxiliary MI estimators, it may be simpler/less finicky to tune than BA/CLUB-based objectives.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently relies on correlation-style debiasing or single-bias fixes, DIR suggests MI-based penalties could capture non-linear dependencies more broadly.
- If GALILEO lacks systematic evaluation on multiple bias types (length + sycophancy + format), DIR’s multi-bias framing is a good evaluation template.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “bias MI” section to related work: contrast Pearson/correlation regularizers vs explicit MI minimization (and note that MI can capture non-linear dependence).
- [ ] Mirror their evaluation axes: create/curate controlled bias settings for length, format, and sycophancy and report both (a) bias reduction and (b) downstream preference/RLHF quality.
- [ ] Consider a pairwise comparative bias regularizer (relative bias attribute between chosen/rejected) as a design pattern.

## Quotes / details to potentially cite

- “Inspired by the information bottleneck (IB), we maximize the mutual information (MI) between RM scores and human preference pairs, while minimizing the MI between RM outputs and biased attributes of preference inputs.”
- “In experiments, we verify the effectiveness of DIR with three types of inductive biases: response length, sycophancy, and format.”
