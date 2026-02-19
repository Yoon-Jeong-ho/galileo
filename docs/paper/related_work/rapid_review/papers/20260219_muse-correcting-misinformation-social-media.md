# Correcting misinformation on social media with a large language model

- Year: 2024
- Venue: arXiv (paper mentions 52 pages; v5 updated 2026-01-11)
- Authors: Xinyi Zhou; Ashish Sharma; Amy X. Zhang; Tim Althoff
- URL: https://arxiv.org/abs/2403.11169
- BibTeX key (if we add it): zhou2024muse
- Tags: misinformation, correction, social-media, multimodal, rag, credibility

## One-sentence takeaway

Muse is a multimodal, credibility-aware retrieval-augmented LLM pipeline that generates grounded, reference-backed corrections to social-media misinformation and is evaluated with detailed rubrics against GPT-4 and community-produced notes.

## What problem does it solve?

- Scaling *high-quality* misinformation correction on social media (identify inaccurate/misleading parts, explain why, and provide credible references) across diverse domains and modalities.
- Addressing common LLM failure modes for correction: outdated knowledge, hallucinated/irrelevant references, and limited handling of images.

## What is the core method / protocol?

- A 3-module system (“Muse” / “MUSE”) for producing correction responses:
  - **LLM response generator**: produces the final correction text.
  - **Hierarchical web retrieval**: generates one or more search queries from the post, retrieves pages, filters by *relevance*, then ranks/filters by **publisher credibility** (factuality + bias ratings) before extracting evidence snippets.
  - **Multimodal integrator**: for posts with images, generates richer image descriptions (explicitly aiming to capture entities + OCR text) so downstream components can verify claims; also does multimodal search / relevance filtering.
- Output format goal: determine whether the content is (fully / partially) inaccurate or potentially misleading; specify which parts; explain with grounded evidence and include reference links.

## What are the key metrics?

- A “comprehensive set of rubrics” spanning:
  - Identification quality (explicitness, accuracy, comprehensiveness, informativeness of what is wrong/misleading vs correct).
  - Generated text quality (relevance, factuality, fluency, coherence, toxicity).
  - Reference quality (reachability, relevance, credibility of cited links).
- Also includes end-user perception / effectiveness: whether corrections improve people’s ability to identify misinformation.

## What are the main results?

- On real-world social media content (evaluated with fact-checking experts), Muse produces higher-quality corrections than baselines:
  - Reported overall improvement vs **GPT-4**: **+37%** (per their aggregate rubric scoring).
  - Reported improvement vs “high-quality responses from social media users” (e.g., Community Notes-style collective responses): **+29%**.
- Generalization claim: works across modalities, domains, and political leanings; includes posts not previously fact-checked online.
- User study: with **988 participants**, Muse corrections improved participants’ ability to correctly identify misinformation by **9.8%**.

## How is this similar to GALILEO?

- If GALILEO targets *grounded, auditable* outputs (especially in high-stakes info settings), Muse is a close neighbor: it combines generation with evidence retrieval and emphasizes reference credibility.
- Emphasis on **evaluation rubrics** beyond simple n-gram similarity (e.g., separating factuality, identification accuracy, and reference credibility), which is often aligned with “trustworthy pipeline” goals.

## How is this different from GALILEO?

- Muse is framed specifically around **social-media correction** and operationalizes a correction as: (1) identify misleading parts, (2) explain, (3) cite sources.
- The retrieval module is explicitly **publisher-credibility-aware** (bias + factuality ratings), which is a specific design choice that may or may not match GALILEO’s evidence selection approach.
- Strong focus on **multimodal** posts (image captioning augmented with celebrity/OCR-like details) and multimodal retrieval filtering.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO has a more principled uncertainty model / calibration, stronger provenance tracking, or stronger guarantees on citation faithfulness, that could be positioned as cleaner than heuristic credibility filtering.
- If GALILEO uses domain-grounded corpora (rather than open-web search), it may offer more reproducibility and fewer risks from web-content drift.

## Where GALILEO is weaker / needs to improve

- If GALILEO currently lacks a credibility-aware retriever (factuality/bias filtering) or strong multimodal claim handling (OCR/entity-rich captioning), Muse suggests concrete gaps.
- If GALILEO evaluation is currently too “one-number” or too automatic, Muse’s rubric set + expert evaluation could be a template.

## Action items for GALILEO (experiments / method / writing)

- [ ] Consider adding (or at least ablation-testing) **credibility-aware retrieval**: rank/filter sources by publisher factuality/bias ratings; report robustness to biased/low-quality sources.
- [ ] Add an ablation: query generation (single vs multiple queries; query decomposition) and its effect on coverage/comprehensiveness.
- [ ] If multimodal is relevant, add an OCR/entity-enhanced image-to-text step and measure improvements on “image with embedded claims”.
- [ ] Borrow rubric dimensions for evaluation section (separate: identification accuracy, explanation factuality, reference credibility/reachability).

## Quotes / details to potentially cite

- “Muse, an LLM augmented with vision-language modeling and web retrieval over relevant, credible sources to generate responses that determine whether and which part(s) of the given content can be misinformed or potentially misleading, and to explain why with grounded references.”
- “Muse outperforms GPT-4 by 37% … and even high-quality responses from social media users by 29%.”
- “A study with 988 participants … demonstrates Muse’s superior performance in increasing people’s ability to correctly identify misinformation by 9.8%.”
