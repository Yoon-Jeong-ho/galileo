# LLMs for Customized Marketing Content Generation and Evaluation at Scale

- Year: 2025
- Venue: KDD LLM4ECommerce Workshop 2025 (arXiv)
- Authors: Haoran Liu et al. (see arXiv submission)
- URL: https://arxiv.org/abs/2506.17863
- BibTeX key (if we add it): liu2025marketingfm-autoeval
- Tags: persuasion-adjacent, marketing, RAG, LLM-as-judge, evaluation, online-ab

## One-sentence takeaway

A retrieval-augmented pipeline (MarketingFM) generates keyword-specific e-commerce ad copy at scale and pairs it with a rules+LLM-judge evaluation stack (AutoEval-Main/Update) validated by human agreement and online A/B tests.

## What problem does it solve?

- Offsite marketing ad copy at large e-commerce sites is often generic (template-based) and poorly aligned with landing pages / product context.
- Manual creation and (especially) manual review of generated ads does not scale.

## What is the core method / protocol?

- **MarketingFM (generation):** retrieval-augmented generation grounded in multiple internal data sources (product/catalog + query/keyword context) to produce **keyword-specific** ad copy; described as task chaining + RAG.
- **Evaluation at scale:**
  - **AutoEval-Main:** combines **rule-based checks** (policy/format/style constraints) + **LLM-as-a-Judge** scoring for higher-level criteria (e.g., relevance, factual alignment).
  - **AutoEval-Update:** an LLM–human loop to keep the evaluation prompt/rubric updated as criteria shift; selectively sample items for human review and use a critic LLM to propose rubric/prompt refinements.
- Validation includes offline human annotation + automated metrics, plus online A/B testing.

## What are the key metrics?

- Online ads: CTR, impressions, CPC.
- Evaluation quality: agreement rate between automated evaluation (LLM-judge+rules) and human reviewers.

## What are the main results?

- In an online experiment, keyword-focused ad copy vs templates achieved up to:
  - **+9% CTR**
  - **+12% impressions**
  - **-0.38% CPC**
- AutoEval-Main reported **89.57% agreement** with human reviewers on large-scale human annotation data.
- AutoEval-Update: critic LLM’s refinement suggestions improved LLM–human agreement (qualitative claim; details not fully captured from the abstract-level skim).

## How is this similar to GALILEO?

- Shares the theme of **evaluation pipelines at scale** where **human review is expensive**, motivating hybrid approaches.
- The **“evaluation prompt/rubric drift”** problem and the need for **dynamic updating** is analogous to keeping safety/robustness evaluation criteria consistent over time.

## How is this different from GALILEO?

- This is **marketing content generation** + evaluation (single-turn-ish ad copy), not multi-turn persuasion/robustness protocols.
- Primary success criteria are **business KPIs** (CTR/CPC) and compliance/quality checks, not truthfulness/consistency under pressure.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO is targeting persuasion/robustness, it likely has a more explicit **adversarial or multi-turn protocol** and clearer causal framing around behavior changes.

## Where GALILEO is weaker / needs to improve

- If GALILEO relies heavily on human labeling, this paper is a reminder that **rules + LLM-judge + selective human audits** can be engineered into a scalable review loop.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider a **2-stage evaluator** pattern: (1) deterministic rule checks (format/policy) + (2) LLM-judge for semantic criteria; report human agreement.
- [ ] Add an “**evaluator maintenance**” subsection: how prompts/rubrics are updated over time, and how you prevent silent criteria drift.
- [ ] If using LLM-judges, consider a lightweight **AutoEval-Update-like** loop (sample + critic report + human thresholding).

## Quotes / details to potentially cite

- “keyword-focused ad copy outperformed templates, achieving up to 9% higher CTR, 12% more impressions, and 0.38% lower CPC” (arXiv abstract).
- “AutoEval-Main achieved 89.57% agreement with human reviewers” (arXiv abstract).
