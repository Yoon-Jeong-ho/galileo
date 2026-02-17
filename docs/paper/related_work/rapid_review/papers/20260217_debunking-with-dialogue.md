# Debunking with Dialogue? Exploring AI-Generated Counterspeech to Challenge Conspiracy Theories

- Slug: debunking-with-dialogue
- Year: 2025
- Venue: WOAH 2025 (Workshop on Online Abuse and Harms), ACL Workshops (per arXiv comments)
- Authors: Mareike Lisker; Christina Gottschalk; Helena Mihaljević
- Links:
  - paper: https://arxiv.org/abs/2504.16604
  - code (if any): https://github.com/HTW-Social-Data-Science/Debunking_with_Dialogue (linked from paper)
- Bibtex: https://arxiv.org/abs/2504.16604 (use arXiv bibtex)

## 1) What problem does it study?
Whether current LLMs can *generate effective counterspeech* (CS) against social-media conspiracy-theory (CT) comments, when guided only by zero-shot prompts describing psychologically-motivated CS strategies.

## 2) Experimental setup (what is being measured?)
- Task(s): Generate counterspeech replies to CT-promoting posts from X/Twitter.
- Perturbation/pressure type: Social persuasion / misinformation-like setting (countering CT claims); also includes hateful and fear-driven framing in the input posts.
- Multi-turn? N (single-turn generation per post; the *application context* is dialogue-like counterspeech, but evaluation is per-comment)
- Metrics:
  - Manual annotation of 456 model responses (152 posts × 3 models) across 12 criteria (mix of binary + Likert), covering: clarity/conciseness, avoiding repetition of harmful content, avoiding stigmatizing terms (e.g., “misinformation”), hate-speech handling (detect/condemn/avoid engaging), fear identification + appropriate empathy, and strategy-specific quality for:
    - Fact-check refutation (Fact)
    - Alternative explanations (Alt)
    - Narrative counter-story (Narr)
    - Critical thinking prompts (Crit)
  - Statistical tests for model comparisons: Friedman + post-hoc Wilcoxon signed-rank; effect sizes via Kendall’s W.
  - Diversity metrics: unique bigrams, Self-BLEU, unique 3-word sentence starts, semantic similarity (SentenceTransformer all-MiniLM-L6-v2).
  - Additional: hate-speech detection F1 on the subset with HS (17 posts).

## 3) Key findings (bullet)
- Across GPT-4o, Llama-3-8B-Instruct, and Mistral-7B-Instruct, outputs are often *generic/repetitive/superficial*, limiting practical utility for NGO settings.
- Hallucination/confabulation is a major barrier: ~10% of fact-checking style outputs include made-up or incorrect “facts/sources/figures”, sometimes subtle and hard to spot.
- Narrative strategy largely fails in the main setting: Narr is used effectively in only ~3/456 cases; even when the prompt is constrained to narrative-only (exploratory GPT-4o run), narratives appear ~60% of the time but are often not effective (mean ≈ 2.82 Likert for narrative quality).
- Models over-acknowledge fear/anxiety: they respond as if fear is present far more often than annotators judged it (annotators <5% vs models 26–52% depending on model).
- Hate-speech handling is weak: HS detection F1 ≈ 0.75 (GPT-4o), 0.69 (Llama 3), 0.30 (Mistral); condemnation quality is low overall.
- Rule-following varies: GPT-4o best at instruction adherence (e.g., separating meta-report with a tag), Llama 3 tends to exceed length limits and is most repetitive; Mistral yields highest lexical/semantic diversity.

## 4) Limitations / threats
- Small dataset (152 posts) and only two CT themes (hate-based “deep state/NWO/globalists” and fear-driven “geo-/bioengineering”).
- No expert “gold” counterspeech references exist for CTs; evaluation relies on manual rubric + single-annotator labeling for much of the data (after limited agreement study + calibration).
- Zero-shot prompting only; findings may differ with few-shot prompting, retrieval grounding, or fine-tuning on curated CS data.
- Single-turn generation; does not directly test multi-turn belief change/revision or longer conversational recovery.

## 5) How it relates to GALILEO
- What we can cite it for:
  - Evidence that prompt-only, zero-shot “persuasion/counterspeech” generation in adversarial belief contexts produces *generic* responses and *non-trivial hallucination rates*, undermining reliability.
  - A concrete *annotation framework + diversity metrics* for conversational intervention outputs (repetition, semantic similarity), which parallels our need to quantify degradation over turns.
- Where we differ (our delta):
  - This is single-turn counterspeech quality/diversity; GALILEO targets *multi-turn robustness under pressure*, including survival/time-to-failure and recovery after flips.
- Direct mapping:
  - Survival ↔ not measured (no time-to-event / multi-turn).
  - TOF ↔ not measured.
  - Recovery ↔ not measured.
  - Neutral Re-asking Control ↔ not applicable.

## 6) Quote-able lines
- “models often generate generic, repetitive, or superficial results.”
- “approximately 10% of outputs contained confabulations … difficult to spot.”
- “Differences between the models were low and mostly not significant.”

## 7) Actions
- [ ] Add to paper: Related work section on LLMs for counterspeech / persuasion interventions in misinformation or CT contexts; use it to motivate grounding + longitudinal robustness metrics.
- [ ] Add to bib
