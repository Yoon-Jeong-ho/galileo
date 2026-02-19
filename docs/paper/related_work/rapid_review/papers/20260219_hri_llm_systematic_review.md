# How Do We Research Human-Robot Interaction in the Age of Large Language Models? A Systematic Review

- Year: 2026
- Venue: CHI 2026 (Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems)
- Authors: Yufeng Wang; Yuan Xu; Anastasia Nikolova; Yuxuan Wang; Jianyu Wang; Chongyang Wang; Xin Tong
- URL: https://arxiv.org/abs/2602.15063
- BibTeX key (if we add it): Wang2026HowDoWeResearchHRIInAgeOfLLMs
- Tags: hri, systematic-review, llm-interaction, embodied-agents, evaluation

## One-sentence takeaway

A PRISMA-style systematic review (86 papers) that proposes a taxonomy (Sense–Interaction–Alignment + design/study dimensions) and summarizes methods/metrics/challenges for LLM-driven HRI.

## What problem does it solve?

- LLM-in-HRI work is rapidly growing but fragmented: papers study different facets (context understanding, social interaction, autonomy, personalization) with inconsistent study designs and evaluation metrics.
- There was limited consolidation of *human-centered* impacts of LLMs in embodied HRI settings (user modeling, alignment with human needs, autonomy levels), making it hard to compare results or identify gaps.

## What is the core method / protocol?

- Systematic literature search following PRISMA.
- Inclusion yields a final corpus of 86 papers.
- Organizes the space with a main conceptual framework:
  - **Contextual Perception and Understanding** (e.g., multimodal physical perception; human-oriented understanding)
  - **Generative and Agentic Interaction** (e.g., social communication; collaborative task co-creation; proactive agency)
  - **Iterative Optimization and Alignment** (e.g., longitudinal personalization/memory; multi-level repair)
- Adds orthogonal “design components and strategies” axes (modality, morphology, autonomy) and “study methods and evaluation strategies” (lab/field/interviews/questionnaires/technical eval; objective vs subjective metrics).

## What are the key metrics?

- This is a review paper; it mainly *surveys* what others use rather than proposing a single metric.
- It explicitly categorizes evaluation into:
  - **Objective metrics** (task performance, success rates, timings, etc.; plus technical measures depending on setup)
  - **Subjective metrics** (user experience, trust, perceived intelligence, comfort, etc.)
- Also reports diversity of study methods (lab experiments, field deployments, interviews, questionnaires, technical evaluations).

## What are the main results?

- LLMs are changing HRI fundamentals: how robots perceive context, generate socially grounded interactions, and maintain alignment with human needs in embodied settings.
- The research landscape is still largely exploratory and heterogeneous: experimental setups, study methods, and metrics vary widely across papers.
- Provides a consolidated overview of application domains and a list of design considerations / challenges (e.g., reliability of LLM-driven understanding, multimodal emotional perception, trust calibration, long-term engagement).

## How is this similar to GALILEO?

- Directly relevant to positioning any LLM-based embodied/interactive agent system: it provides shared vocabulary for **context understanding**, **agentic interaction**, and **alignment over time**.
- The review’s “Iterative optimization and alignment” section includes **longitudinal personalization and memory**, which overlaps with GALILEO-style long-horizon user adaptation.
- Useful for the *related work* and *evaluation* sections: it inventories typical HRI study methodologies and metrics used for LLM-driven interaction.

## How is this different from GALILEO?

- This is not a new model/system; it is a taxonomy + synthesis of 86 papers.
- It does not provide a concrete algorithmic contribution (e.g., memory architecture, planning, preference learning), nor a unified benchmark.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO provides a specific end-to-end system and evaluation protocol, it can be more actionable than the survey’s high-level categorization.
- GALILEO can contribute standardized evaluation tasks/metrics for long-term alignment/personalization that the review highlights as currently inconsistent.

## Where GALILEO is weaker / needs to improve

- The review indicates the community expects careful *human-centered* evaluation (trust, comfort, long-term engagement, repair). If GALILEO is currently mostly technical/offline, it may under-address these dimensions.

## Action items for GALILEO (experiments / method / writing)

- [ ] Use the Sense–Interaction–Alignment framing to structure GALILEO related work (especially if we claim improvements in interaction and alignment over time).
- [ ] Ensure the evaluation section clearly separates objective vs subjective metrics (even if we only do objective now, explicitly justify and discuss subjective/HRI limitations).
- [ ] Add a “limitations + future work” paragraph aligned with the review’s highlighted challenges: reliability, trust calibration, and long-term engagement.

## Quotes / details to potentially cite

- “conducted a systematic literature search following the PRISMA guideline, identifying 86 articles” (abstract).
- Key framing: LLMs reshape HRI via (i) contextual perception/understanding, (ii) generative/agentic interaction, (iii) iterative optimization/alignment (paper’s section structure).
- Venue/DOI: CHI ’26; https://doi.org/10.1145/3772318.3790920 (from arXiv HTML page).
