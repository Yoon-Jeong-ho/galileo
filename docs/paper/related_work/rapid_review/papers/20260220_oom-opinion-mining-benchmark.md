# Can Large Language Models be Effective Online Opinion Miners?

- Year: 2025
- Venue: EMNLP 2025 (Main)
- Authors: Ryang Heo, Yongsik Seo, Junseong Lee, Dongha Lee
- URL: https://arxiv.org/abs/2505.15695
- BibTeX key (if we add it): heo2025oom
- Tags: opinion-mining, benchmarks, information-extraction, llm-evaluation, llm-as-judge

## One-sentence takeaway

Introduces OOMB, a benchmark for opinion mining on realistic online content that jointly evaluates structured tuple extraction and higher-level insight generation, finding LLMs weak at strict extraction but comparatively better at abstractive synthesis.

## What problem does it solve?

- Existing opinion mining benchmarks often use simplified inputs (short reviews / preprocessed dialogs) and focus on extraction-only settings, which do not reflect the complexity of real online opinion streams (long-form blogs, threaded discussions, slang/implicit sentiment).
- Lack of an evaluation setup that tests both (a) extracting structured opinion units and (b) producing marketer-style topical insights from messy online content.

## What is the core method / protocol?

- Dataset/benchmark: **OOMB (Online Opinion Mining Benchmark)** with content from multiple sources (blogs, review sites, Reddit, YouTube).
- Dual-layer annotation per content:
  - **(entity, feature, opinion)** tuples (feature can be implicit → labeled as `NULL`).
  - **Opinion-centric insight**: a concise (3–5 line) topic-organized summary “from a marketing manager’s perspective.”
- Human-in-the-loop annotation:
  - LLM generates candidates (multiple rounds); humans verify, de-hallucinate, and add missing tuples.
  - For insights, multiple candidate insights are generated and humans select/refine the best.
- Two tasks:
  - **FOE** (Feature-centric opinion extraction): predict set of (e,f,o) tuples.
  - **OIG** (Opinion-centric insight generation): generate the topic-level insight text.

## What are the key metrics?

FOE (tuple extraction):
- **Exact Match (EM)**: strict component match.
- **Relaxed Match (RM)** with threshold 0.7:
  - Lexical RM via token overlap (difflib).
  - Semantic RM via sentence transformer similarity.
- **Contextual Match (CM)**: LLM-based matching (uses GPT-4o as a judge to decide tuple matches), intended to approximate human judgment.

OIG (insight generation):
- Reference-based: ROUGE-1/2/L, BERTScore, A3CU.
- Reference-free: LLM-as-a-judge on criteria like faithfulness, coverage, specificity, insightfulness, intent, fluency.

## What are the main results?

- FOE is hard: even strong models achieve low F1 under EM/RM; CM improves scores and correlates better with human judgments (they report CM having much higher correlation than EM/RM).
- Common extraction failure mode: models paraphrase or “reinterpret” spans instead of extracting verbatim, and may substitute related concepts (e.g., “engine” vs “power”) or hallucinate.
- OIG is comparatively stronger: models produce fluent, faithful insights, but still struggle with implicit intent and deeper “insightfulness.”
- Providing structured tuples as additional input improves insight generation metrics and perceived coverage/insightfulness (with some tradeoff in fluency/intent).

## How is this similar to GALILEO?

- Evaluates LLM behavior on **realistic, messy online text** (including threaded discussions), which overlaps with GALILEO’s emphasis on robustness in realistic interaction environments.
- Uses **LLM-as-a-judge** style evaluation (CM for tuple matching; judge rubric for generation), relevant to GALILEO if it uses learned/judge-based metrics.
- Highlights a recurring LLM reliability issue: tendency to transform inputs rather than adhere to exact extraction/spec constraints.

## How is this different from GALILEO?

- Focus is opinion mining (entity/feature/opinion extraction + marketing-style insight summaries), not multi-turn robustness, persuasion/sycophancy, belief revision, or stability/drift controls.
- Primary emphasis is benchmark construction + offline evaluation, not interactive multi-round protocols.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides controlled multi-turn setups and explicit stability/drift measurements, it covers failure modes (drift, susceptibility, pressure) not addressed by OOMB.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies on strict-match metrics for structured outputs, OOMB’s findings suggest these can undervalue “near-miss but semantically correct” outputs; may need complementary judge-based matching.
- If GALILEO benchmarks are mostly short/clean inputs, OOMB motivates adding **long, dense, noisy** online content variants.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding a “noisy online content” slice (long-form + threaded) to stress-test GALILEO robustness claims.
- [ ] Consider a dual-metric scheme for structured outputs: strict exact match + a contextual/judge-based match to better align with human utility.
- [ ] When discussing evaluation, cite OOMB as evidence that extraction tasks can look artificially poor under strict span matching due to paraphrase/substitution behaviors.

## Quotes / details to potentially cite

- OOMB includes both structured tuples and an “opinion-centric insight” per content to evaluate extractive + abstractive capabilities.
- CM (LLM-based tuple matching) is reported as substantially better aligned with human judgment than EM/RM (they provide correlation table showing CM highest).
- Noted extraction failure pattern: models often produce semantically related but non-identical spans, swap related concepts, or hallucinate when asked to extract spans as-is.
