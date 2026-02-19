# RoParQ: Paraphrase-Aware Alignment of Large Language Models Towards Robustness to Paraphrased Questions

- Year: 2025
- Venue: arXiv (cs.CL)
- Authors: Minjoon Choi
- URL: https://arxiv.org/abs/2511.21568
- BibTeX key (if we add it): roparq2025choi
- Tags: robustness, paraphrase, consistency, benchmark, alignment, SFT, MCQA

## One-sentence takeaway

RoParQ proposes a paraphrase-focused MCQA benchmark plus an “across-paraphrase accuracy variance” metric (XParaCon), and shows paraphrase-aware SFT can substantially improve paraphrase robustness/consistency.

## What problem does it solve?

- LLMs can answer a question correctly in one phrasing but fail on semantically-equivalent paraphrases, indicating reliance on superficial cues and hurting reliability.
- Existing evaluations often measure single-prompt accuracy; they do not directly quantify *cross-paraphrase consistency*.

## What is the core method / protocol?

- **Benchmark construction (RoParQ):**
  - Start from standard multiple-choice QA sources (mentions MMLU, ARC, CommonsenseQA, MathQA).
  - For each original question, generate **two paraphrases** using proprietary models (Gemini 2.5 Flash Lite; Claude 3.5 Sonnet), yielding 3 variants total: {original, gemini paraphrase, claude paraphrase}.
  - To reduce sensitivity to *choice ordering*, create **8 random permutations** of answer choices per question variant.
  - Define “**perfectly correct**” for a question variant as: correct across **all 8** choice permutations.
  - Use an open-source **judge LLM (Llama-3.1-8B-Instruct)** to categorize each example as consistently confident / consistently unconfident / **inconsistently confident** based on which of the 3 variants are perfectly correct.
  - Keep only the **inconsistent-confidence** cases (the failure mode of interest).

- **Metric (XParaCon):**
  - Robustness is quantified via the **standard deviation of accuracies across the paraphrase variants** (the paper positions higher “consistency” as better; double-check sign when citing).

- **Alignment method:**
  - A **reasoning-based, paraphrase-aware SFT** strategy intended to train semantic invariance across paraphrases.

## What are the key metrics?

- Accuracy on MCQA.
- **XParaCon (Cross-Paraphrase Consistency):** standard deviation of accuracy across the 3 paraphrase variants.
- “Perfectly correct” rate under 8 choice permutations (used for filtering/labeling confidence consistency).

## What are the main results?

- Robustness tends to improve with model scale, but paraphrase inconsistency persists.
- Paraphrase-aware alignment via SFT improves consistency; example stated in the HTML version:
  - Llama-3.1-8B-Instruct XParaCon **rises from 2.186 to 2.629** after alignment (exact interpretation of direction should be verified before quoting in GALILEO).
- Claims lightweight fine-tuned models can reach consistency comparable to larger pre-trained models.

## How is this similar to GALILEO?

- Same broad theme: **robustness under semantically-equivalent perturbations** (here: paraphrases; in GALILEO: multi-round robustness / survival / TOF under iterative probing).
- Emphasizes that single-shot benchmark accuracy can hide brittleness; robustness should be measured explicitly.

## How is this different from GALILEO?

- Task setting is **closed-book multiple-choice QA**, not GALILEO’s multi-round interaction/protocol.
- Uses **paraphrase variants + choice-order permutations** rather than adversarial rounds.
- Proposes **stddev-of-accuracies** as robustness signal, rather than survival / turn-of-failure / flip analyses.
- Benchmark construction relies on **proprietary paraphrase generators**, which may be a reproducibility concern (though they release dataset/code).

## Where GALILEO is stronger / cleaner (if true)

- GALILEO’s survival/TOF framing can be more **mechanistic and failure-traceable** (what changed at which round), whereas XParaCon is a scalar dispersion summary.
- GALILEO can avoid dependence on proprietary paraphrase generation by using controlled transformations or open generators (if we choose).

## Where GALILEO is weaker / needs to improve

- GALILEO should clearly position robustness beyond paraphrase consistency and explain why multi-round stress tests capture additional real-world brittleness.
- If we lack a “prompt-variant robustness” baseline, reviewers could ask why we don’t evaluate paraphrase invariance; RoParQ is a natural comparator/citation.

## Action items for GALILEO (experiments / method / writing)

- [ ] Related work: cite RoParQ as paraphrase-robustness benchmark + metric; contrast with multi-round robustness and survival/TOF framing.
- [ ] Consider adding a *small* paraphrase-perturbation ablation: run GALILEO tasks with 2–3 paraphrases of the initial prompt and report survival/TOF sensitivity.
- [ ] If adding such ablation, define a GALILEO-friendly metric (e.g., variance of survival at fixed round r across paraphrases) to connect to XParaCon.

## Quotes / details to potentially cite

- “Large Language Models (LLMs) often exhibit inconsistent behavior when answering paraphrased questions…” (Abstract)
- “We introduce RoParQ … constructed by paraphrasing questions from established benchmarks … and selectively retaining examples that elicit inconsistent confidence from a judge model.” (Abstract/Intro)
- “We further propose XParaCon … measuring the standard deviation of accuracies across question variants.” (Abstract)
- Protocol detail worth citing: **8 random permutations of choices** and “perfectly correct” iff correct under all permutations.
