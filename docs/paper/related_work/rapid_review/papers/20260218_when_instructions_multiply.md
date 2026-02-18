# When Instructions Multiply: Measuring and Estimating LLM Capabilities of Multiple Instructions Following

- Year: 2025
- Venue: EMNLP 2025 (accepted)
- Authors: Keno Harada; Yudai Yamazaki; Masachika Taniguchi; Edison Marrese-Taylor; Takeshi Kojima; Yusuke Iwasawa; Yutaka Matsuo
- URL: https://arxiv.org/abs/2509.21051
- BibTeX key (if we add it): harada2025when
- Tags: instruction-following, multi-instruction, evaluation, benchmarking, performance-estimation

## One-sentence takeaway

They introduce controlled multi-instruction benchmarks for text + code and show instruction-following accuracy degrades with more constraints, while a simple logistic regression on instruction count can estimate performance on unseen combinations with ~10% error.

## What problem does it solve?

- Real prompts often contain multiple simultaneous constraints (formatting, style, length, etc.), but existing instruction-following benchmarks confound instruction-count with changing task descriptions or rely on unstable LLM-judge evaluation.
- Exhaustively evaluating all combinations of multiple instructions is combinatorially expensive; practitioners need ways to estimate performance without testing every combination.

## What is the core method / protocol?

- **Benchmarks** (controlled design: same base task description, vary number of extra instructions):
  - **ManyIFEval**: text generation benchmark extended from IFEval, up to **10** instructions.
  - **StyleMBPP**: code generation benchmark extended from MBPP, up to **6** instructions, with style constraints.
- **Evaluation**: emphasizes **rule-based / programmatically verifiable** checking of instruction satisfaction (and for StyleMBPP, also rule-based verification of task description correctness).
- **Estimation models** to predict multi-instruction success without enumerating combinations:
  - naive estimators; beta-binomial; **logistic regression** using instruction count as an explanatory variable.

## What are the key metrics?

- Instruction-following performance as a function of **number of instructions** (success rate / accuracy under rule-based checks).
- Generalization to **unseen instruction combinations**.
- Estimation error for predicted performance (they report ~**10%** error for logistic regression; also report **MAE ~0.03 ± 0.04** when predicting 10-instruction performance from training data up to 9 instructions).
- Sample efficiency: how many evaluated prompts are needed to estimate performance reliably (reported: ~500 ManyIFEval / ~300 StyleMBPP).

## What are the main results?

- Across **10 LLMs**, performance **consistently (and “drastically”) degrades** as instruction count increases.
- A **logistic regression model** using just instruction count can predict multi-instruction performance for unseen combinations with roughly **10%** error.
- Modest sample sizes (order of hundreds) can suffice for performance estimation under varying instruction counts.

## How is this similar to GALILEO?

- If GALILEO involves **multi-constraint generation/evaluation** (e.g., satisfying several requirements simultaneously), this paper’s framing and controlled measurement of “constraint count” effects is directly relevant.
- The regression-based approach provides a potential **cheap performance-estimation tool** when the evaluation space is combinatorial.

## How is this different from GALILEO?

- Focuses on **benchmark construction + evaluation/estimation**, not on training or inference-time methods to improve multi-instruction satisfaction.
- Centers instruction-following with **rule-based verifiers**, rather than subjective preference/judge-based scoring.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO proposes a method to *improve* adherence to multiple constraints (not just measure it), it goes beyond this paper’s evaluation/estimation contribution.

## Where GALILEO is weaker / needs to improve

- If GALILEO claims robustness to multiple constraints, it may need a **controlled “vary only instruction-count”** analysis like ManyIFEval/StyleMBPP to support that claim.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a small related-work paragraph: “multi-instruction following degrades with instruction count; controlled benchmarks (ManyIFEval/StyleMBPP) quantify this; simple count-based logistic regression can estimate performance under combinatorial instruction sets.”
- [ ] Consider adding an analysis figure/table showing GALILEO performance vs **#constraints** with the task fixed (mirroring their controlled design rationale).
- [ ] If relevant, explore whether a simple count-based model predicts GALILEO’s success rate across constraint sets (as a sanity check / baseline).

## Quotes / details to potentially cite

- “We introduce two specialized benchmarks… ManyIFEval… up to ten instructions… StyleMBPP… up to six instructions.”
- “Performance consistently degrades as the number of instructions increases.”
- “A logistic regression model using instruction count… can predict performance… with approximately 10% error, even for unseen instruction combinations.”
- Sample size claim: “500 for ManyIFEval and 300 for StyleMBPP… sufficient for performance estimation.”
