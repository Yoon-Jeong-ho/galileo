# C-MTCSD: A Chinese Multi-Turn Conversational Stance Detection Dataset

- Year: 2025
- Venue: WWW Companion ’25
- Authors: Fuqiang Niu et al.
- URL: https://arxiv.org/abs/2504.09958
- BibTeX key (if we add it): Niu2025CMTCSD
- Tags: multi-turn, stance-detection, chinese, dataset, benchmark, zero-shot

## One-sentence takeaway

A large, carefully-annotated Chinese multi-turn conversational stance detection benchmark (24k instances from Weibo) that exposes steep performance degradation with conversation depth and challenging implicit/implicit-target stance cases.

## What problem does it solve?

- Prior stance detection datasets/methods are dominated by English and single-utterance settings, while real social-media stance is often conversational (multi-turn) and Chinese.
- Existing Chinese conversational stance datasets are small; this limits reliable evaluation and progress tracking.

## What is the core method / protocol?

- Dataset construction from Sina Weibo conversations around 5 targets (2 tech: iPhone 15, Apollo Go; 3 social issues: non-marriage doctrine, naked resignation, pre-made meals).
- Multi-turn conversation depth up to 6; they provide instances across depths (1–6), with a large mass at depth 3.
- Annotation: stance labels {Against, Favor, None}; each instance annotated by at least 2 annotators; second-round re-annotation for low-agreement cases; high reported agreement (avg. kappa ~0.93).
- Benchmarking across:
  - Traditional neural baselines (e.g., TAN, CrossNet)
  - Pretrained LMs (BERT/RoBERTa/XLNet on Chinese)
  - Conversation-aware stance models (e.g., Branch-BERT, GLAN)
  - LLM prompting approaches (incl. GPT-4) for zero-shot.

## What are the key metrics?

- F_avg: mean F1 over the “Against” and “Favor” classes (computed per target, then averaged).

## What are the main results?

- Scale: 24,264 instances; claimed 4.2× larger than prior Chinese conversational stance dataset (CANT-CSD).
- Depth distribution: substantial long-context portion (reported ~27.7% with depth > 3); depth up to 6.
- Zero-shot is hard:
  - Best reported average zero-shot performance: GPT-4 with F_avg = 64.07.
  - Traditional models struggle on implicit stance; paper highlights many are <50 F1 in some challenging settings.
- Performance generally degrades as conversation depth increases (they report depth-bucketed results).

## How is this similar to GALILEO?

- Highlights the importance of modeling *interaction structure / conversational context* rather than isolated text.
- Provides an evaluation lens for robustness under longer, branched conversational histories and implicit references—likely aligned with GALILEO’s goals if GALILEO targets grounded conversational understanding.

## How is this different from GALILEO?

- Primarily a dataset + benchmark paper (and baseline comparisons), not a new modeling framework aimed at general reasoning/grounding.
- Focused on stance detection (Against/Favor/None) in Chinese Weibo threads over a fixed set of targets; not a general multi-task conversational benchmark.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO uses controlled data generation, cleaner provenance, or richer supervision signals, it may avoid platform-specific artifacts (Weibo-specific slang, moderation effects, etc.).
- If GALILEO supports broader tasks beyond stance, it can position stance as one slice of conversational understanding.

## Where GALILEO is weaker / needs to improve

- If GALILEO lacks large-scale *real* multi-turn social-media-like conversational data with deep contexts and implicit target references, C-MTCSD demonstrates a gap.
- If GALILEO evaluation does not explicitly stratify by depth/turn count, it may miss the degradation effect this paper reports.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an evaluation stratified by conversation depth (e.g., 1–2 / 3–4 / 5–6 turns) and report degradation curves.
- [ ] Create/curate test slices emphasizing implicit stance and implicit target reference (coreference-heavy) cases.
- [ ] Consider including a “None/neutral/irrelevant” label (or an abstain option) if stance-like tasks appear in GALILEO.
- [ ] In related work, cite this as evidence that even strong LMs struggle in zero-shot conversational stance and that depth matters.

## Quotes / details to potentially cite

- “C-MTCSD … comprising 24,264 carefully annotated instances from Sina Weibo … 4.2 times larger than the only prior Chinese conversational stance detection dataset.”
- “Even our best-performing LLM-based approach achieves an F1 score of only 64.07% in the challenging zero-shot setting, while performance consistently degrades with increasing conversation depth.”
- Targets include: iPhone 15, Apollo Go, non-marriage doctrine, naked resignation, pre-made meals; depth up to 6.
