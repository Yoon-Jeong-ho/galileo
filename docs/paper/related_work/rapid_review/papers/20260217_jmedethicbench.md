# JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese Large Language Models

- Year: 2026
- Venue: arXiv
- Authors: Junyu Liu; Zirui Li; Qian Niu; Zequn Zhang; Yue Xun; Wenlong Hou; Shujun Wang; Yusuke Iwasawa; Yutaka Matsuo; Kan Hatakeyama-Sato
- URL: https://arxiv.org/abs/2601.01627
- BibTeX key (if we add it): liu2026jmedethicbench
- Tags: medical, safety, multi-turn, benchmark, jailbreak

## One-sentence takeaway

A Japanese, guideline-grounded, multi-turn medical-safety benchmark shows safety degrades across turns and that domain-specialized medical LLMs can be more jailbreak-vulnerable than general commercial models.

## What problem does it solve?

- Existing medical safety benchmarks are (a) English-centric and (b) largely single-turn, missing the risk of *conversational escalation* in realistic multi-turn consultations.
- Need a benchmark tied to concrete medical-ethics guidance for Japan rather than coarse harm categories.

## What is the core method / protocol?

- Construct JMedEthicBench grounded in **67 guidelines** from the Japan Medical Association.
- Generate **~50k+ adversarial multi-turn conversations** using **7 automatically discovered jailbreak strategies**.
- Evaluate **27 models** with a **dual-LLM scoring protocol** (LLM-as-judge style) to rate safety over turns; also consider helpfulness (they mention a licensing-exam question set for helpfulness).

## What are the key metrics?

- Turn-aware safety scoring; they report **safety score decline across conversation turns**.
- Safety pass rate example definition (from intro/fig caption): % of conversations where *any turn* is above a safety threshold.
- Cross-lingual comparison (Japanese vs English versions) to test whether vulnerabilities persist across language.

## What are the main results?

- Commercial models are comparatively robust on this benchmark, while **medical-specialized models are more vulnerable** to adversarial multi-turn pressure.
- **Safety declines with turn index** (reported median drop **9.5 → 5.0**, with **p < 0.001**).
- Cross-lingual evaluation suggests the vulnerability pattern persists across Japanese/English, pointing to alignment limitations not purely language-specific.

## How is this similar to GALILEO?

- Shared focus on **multi-turn evaluation** where behavior drifts/weakens over turns rather than a single prompt.
- Emphasizes *protocol + measurement* (multi-turn threat surface) rather than only static capability scores.

## How is this different from GALILEO?

- Targets **medical safety / ethics** (domain safety + jailbreaking) rather than general conversational stability/truthfulness.
- Uses a large synthetic/adversarial conversation generation pipeline + LLM judges; GALILEO may emphasize different constructs (e.g., drift vs evidence-driven revision) and different ground-truthing.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO explicitly separates *drift* from *legitimate belief update* with controlled evidence conditions, that’s a cleaner causal handle than generic jailbreak adversaries.

## Where GALILEO is weaker / needs to improve

- Could underweight **domain-specific safety** and the empirical fact that **turn-by-turn safety erosion** is substantial; may need stronger turn-indexed “time-to-failure” style reporting.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add an analysis view that plots score vs turn and reports median/quantiles + a “time-to-violation” statistic.
- [ ] Consider a small cross-lingual or domain-shift slice to test whether failure modes persist across surface form.
- [ ] In related work, cite this as evidence that multi-turn interactions are a distinct safety threat surface and that specialization can weaken safety.

## Quotes / details to potentially cite

- “...contains over **50,000 adversarial conversations** generated using **seven automatically discovered jailbreak strategies**.” (abstract)
- “...safety scores decline significantly across conversation turns (**median: 9.5 to 5.0**, *p* < 0.001).” (abstract)
- Grounding: “...based on **67 guidelines** from the **Japan Medical Association**...” (abstract/introduction)
