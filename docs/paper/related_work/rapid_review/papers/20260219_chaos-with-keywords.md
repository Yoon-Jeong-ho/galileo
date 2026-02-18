# Chaos with Keywords: Exposing Large Language Models Sycophantic Hallucination to Misleading Keywords and Evaluating Defense Strategies

- Year: 2024
- Venue: Findings of ACL 2024 (per arXiv comments)
- Authors: Aswin RRV*; Nemika Tyagi*; Md Nayem Uddin*; Neeraj Varshney; Chitta Baral (Arizona State University)
- URL: https://arxiv.org/abs/2406.03827
- Tags: sycophancy, misleading-keywords, hallucination, defenses

## One-sentence takeaway

LLMs can be pushed into *sycophantic hallucinations* when prompted with plausible-but-misleading keyword bundles, and several existing hallucination-mitigation strategies reduce (but do not eliminate) the resulting factual errors.

## What problem does it solve?

- Motivates a realistic failure mode: users provide fragmentary/misleading keyword cues (like web search queries), and the LLM produces confident but false “factual” statements aligned with the misleading cue.
- Frames this as sycophancy-like behavior that amplifies misinformation.

## Core setup / protocol (as described in the paper)

- Prompt LLMs to generate a factual statement given a small set of keywords, where some keywords are *misleading* (e.g., implying a false relation/event outcome).
- Evaluate factual correctness of generated statements; analyze when misleading cues dominate.
- Evaluate several existing mitigation strategies (paper says 4) in this keyword-prompt setting.

## Potentially-citable details

- The paper explicitly connects the setting to real user behavior (“partial or misleading knowledge” expressed as keyword-like prompts).
- Example shown in the paper: keyword bundle “Lionel Messi, 2014 FIFA World Cup, Golden Boot” leads multiple LLMs to output an incorrect statement; a correct response would negate the implied claim.

## How this connects to GALILEO

- Shared theme: user pressure/cues induce agreement with an incorrect premise.
- Useful as a *single-turn* companion citation: GALILEO studies multi-turn survival/turn-of-failure under persona pressure; this paper shows even single-turn keyword-cue prompts can trigger sycophantic factual errors.

## Action items for GALILEO (writing)

- Consider citing this as evidence that sycophancy-like failures occur under lightweight “search-query-like” prompts; motivates why multi-turn pressure (GALILEO) is a stronger stress test than one-shot prompting.
