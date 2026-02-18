# Automatically Finding Reward Model Biases

- Year: 2026
- Venue: arXiv (submitted to ICML per paper header)
- Authors: Zifan Wang et al.
- URL: https://arxiv.org/abs/2602.15222
- BibTeX key (if we add it): wang2026automatically
- Tags: reward-model, bias, diagnostics, black-box, evolutionary-search

## One-sentence takeaway

A black-box “evolutionary” LLM-in-the-loop audit that automatically proposes natural-language reward-model bias hypotheses and validates them via counterfactual rewrite pairs, surfacing issues like preference for extra whitespace or hallucinated content.

## What problem does it solve?

- Reward models (RMs) used in RLHF can systematically prefer undesirable attributes (length, format artifacts, hallucinations, sycophancy, etc.), but today these are often found late (after downstream model optimization) or as one-off adversarial examples.
- The paper targets *automatic discovery* of *generalizable, natural-language-described* RM biases directly from RM preferences, without training/optimizing a policy against the RM.

## What is the core method / protocol?

- Define an “attribute” A as a natural-language feature plus a binary classifier A(x)∈{0,1} (implemented via LLM judge or simple regex for some cases).
- Create *counterfactual response pairs* for the same prompt that differ primarily in whether the attribute is present:
  - Sample a response y, then use an LLM rewriter to minimally edit it into y_{A=1} and y_{A=0}.
  - Use multiple different rewrite models to reduce correlated artifacts.
- Quantify:
  - RM bias strength R(A) = E[ R(x, y_{A=1}) − R(x, y_{A=0}) ]
  - LLM-judge “bias winrate” J(A) = E[ judge prefers y_{A=1} over y_{A=0} ]
- Call A a problematic RM bias when RM prefers it (R(A) > 0) but a strong LLM judge disfavors it (J(A) < 0.5).
- Search procedure (“evolutionary loop”):
  - Start with candidate bias descriptions proposed by an LLM from observed trends.
  - Evaluate candidates via the counterfactual-pair metrics.
  - Keep the promising ones and have an LLM propose/refine variations; repeat.
  - Compare against flat best-of-N hypothesis generation.

## What are the key metrics?

- R(A): average RM score delta between attribute-present vs attribute-absent rewrites.
- J(A): LLM-judge preference winrate (attribute-present vs absent).
- Evidence of search effectiveness: evolutionary iteration vs best-of-N; recall checks via synthetically injected biases.

## What are the main results?

- Recovers known classes of RM biases and surfaces novel ones on a strong open-weight RM (Skywork-V2-8B), including:
  - RM preferring responses with *redundant spacing/whitespace*.
  - RM preferring responses with *hallucinated content* (consistent with prior reports).
- Evolutionary iteration provides better results than simple best-of-N candidate generation (as reported).
- Synthetic bias injection suggests the pipeline has reasonable recall for planted biases.

## How is this similar to GALILEO?

- Treats audit as *black-box evaluation* with carefully constructed counterfactuals to isolate “cheatable” or spurious features.
- Uses an explicit “two-objective” framing: maximize RM preference while minimizing a proxy for human preference (LLM judge), akin to searching for misalignment between a learned scorer and a target notion of quality.

## How is this different from GALILEO?

- Primary target is *reward model auditing* (finding RM biases), not end-to-end policy training or broader interactive alignment objectives.
- Strong reliance on LLMs for: (i) proposing bias hypotheses and (ii) rewriting to create counterfactual pairs.
- Focuses on *natural-language-described attributes* and their measurement via rewrite-based counterfactuals, rather than (e.g.) representation-level interpretability or direct policy-level exploitation.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO can produce more controlled counterfactual generation (e.g., stronger minimality guarantees / automated checks), it may reduce confounding from rewrite artifacts.
- If GALILEO supports human-in-the-loop validation or robust uncertainty estimates, it can better distinguish “true RM bias” vs “judge/rewriter idiosyncrasy.”

## Where GALILEO is weaker / needs to improve

- Automated discovery pipelines like this suggest GALILEO may need a systematic “bias hypothesis generator + refinement loop” rather than only manual checklists.
- Need robust tooling to detect non-semantic artifacts (spacing, punctuation, formatting tricks) that can dominate learned scorers.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a “formatting artifact” audit suite: whitespace padding, bullet formatting, markdown styling, etc., to test whether any internal evaluators/scorers can be gamed.
- [ ] Implement an iterative hypothesis refinement loop for evaluator bias discovery (start from coarse hypotheses, mutate/refine based on metric feedback).
- [ ] Include a counterfactual-pair minimality check (e.g., edit distance / semantic similarity constraints) to ensure attribute isolation.

## Quotes / details to potentially cite

- Abstract framing: RMs can reward spurious attributes (length/format/hallucination/sycophancy), and the paper proposes an LLM-driven iterative pipeline to *automatically find reward model biases*.
- Example bias: Skywork-V2-8B “mistakenly favors responses with redundant spacing” and sometimes “responses with hallucinated content.”
