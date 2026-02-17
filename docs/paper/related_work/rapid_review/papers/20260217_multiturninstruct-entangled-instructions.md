# Can Language Models Follow Multiple Turns of Entangled Instructions?

- Year: 2025
- Venue: Findings of EMNLP 2025 (arXiv)
- Authors: Chi Han, Xin Liu, Haodong Wang, Shiyang Li, Jingfeng Yang, Haoming Jiang, Zhengyang Wang, Qingyu Yin, Liang Qiu, Changlong Yu, Yifan Gao, Zheng Li, Bing Yin, Jingbo Shang, Heng Ji
- URL: https://arxiv.org/abs/2503.13222
- BibTeX key (if we add it): Han2025MultiTurnInstruct
- Tags: multi-turn, instruction-following, conflicting-instructions, benchmark, privacy

## One-sentence takeaway

MultiTurnInstruct is a human-in-the-loop benchmark (~1.1k multi-turn conversations) that probes whether LLMs can **retrieve, track/reason over, and resolve conflicts among** evolving instructions, revealing capability trade-offs (e.g., strong memorization but weak selective withholding/privacy).

## What problem does it solve?

- Real deployments require models to follow **multiple instructions over time** that may be entangled (overlapping) or conflicting (preference vs privacy vs prioritization), not just single-turn “do X”.
- Existing instruction-following evaluations under-test: (i) long-range instruction retrieval, (ii) cross-turn reasoning about instructions, (iii) explicit conflict resolution.

## What is the core method / protocol?

- Build **MultiTurnInstruct**: ~1.1k multi-turn conversations, curated with a human-in-the-loop process.
- Evaluate LLMs on 3 escalating levels:
  1) **Retrieval**: recall facts/preferences specified earlier in the instruction history.
  2) **Tracking + reasoning**: maintain and apply instruction constraints across turns.
  3) **Conflict resolution**: handle intersecting/conflicting instructions (priorities, exceptions, etc.).
- Organize tasks into **nine capability categories** (the paper groups them across statics/dynamics, reasoning, multitasking).

## What are the key metrics?

- Task-specific success / accuracy for the above capability categories.
- The paper notes that strong **memorization** (e.g., BLEU-based overlap on recall-oriented tasks) does not imply correct behavior on privacy/selective withholding.

## What are the main results?

- GPT-family models show **strong memorization**, but are **less effective at privacy-protection** tasks that require selectively withholding information.
- Larger models generally show **better reasoning**, yet still struggle at **resolving conflicting instructions**.
- The failures are not just “forgot the instruction”: models can score highly on memorization-style metrics while still failing to integrate related instructions when they must be combined or traded off.

## How is this similar to GALILEO?

- Shared emphasis on **multi-turn robustness** where later turns test whether earlier constraints persist.
- Highlights that “multi-turn capability” decomposes into subskills (retrieval vs reasoning vs conflict resolution), aligning with GALILEO’s need to separate *benign updating* from *unwanted drift*.

## How is this different from GALILEO?

- Focuses on **instruction-following** (preferences/privacy/prioritization), not specifically persuasion/social-pressure-induced drift or sycophancy.
- Primary outcomes are per-category task performance, rather than time-to-failure / recovery trajectories under pressure operators.

## Where GALILEO is stronger / cleaner (if true)

- If GALILEO includes paired conditions (neutral vs pressure vs evidence), it can more cleanly isolate **pressure-driven drift** from legitimate updates; MultiTurnInstruct is broader instruction competence rather than drift attribution.
- GALILEO-style trajectory metrics (time-to-flip, recovery rate) would add dynamics beyond category-wise success.

## Where GALILEO is weaker / needs to improve

- MultiTurnInstruct underscores that **privacy / selective withholding** is a distinct multi-turn competence where even strong memorization can be misleading; GALILEO may need an explicit “withhold vs reveal” axis if we make claims about instruction-following robustness.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add (or cite) an explicit decomposition of multi-turn instruction-following into **retrieval / tracking+reasoning / conflict-resolution**, to motivate why single aggregate metrics can be insufficient.
- [ ] Consider a small ablation/operator that tests **selective withholding** across turns (privacy constraint) to complement persuasion/pressure operators.

## Quotes / details to potentially cite

- Abstract framing: real settings require “consistency across multiple instructions over time … privacy, personal preferences, and prioritization” and must “integrate multiple turns and carefully balance competing objectives when instructions intersect or conflict.”
- Main finding (abstract): “trade-off between different capabilities … GPT models demonstrate superior memorization … reduced effectiveness in privacy-protection … Larger models … still struggle with resolving conflicting instructions … gaps cannot be attributed solely to information loss.”
