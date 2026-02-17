# User-Assistant Bias in LLMs

- Year: 2026 (arXiv v2)
- Venue: arXiv
- Authors: Xu Pan; Jingxuan Fan; Zidi Xiong; Ely Hahami; Jorin Overwiening; Ziqian Xie
- URL: https://arxiv.org/abs/2508.15815
- BibTeX key (if we add it): pan2026userassistantbias
- Tags: sycophancy-adjacent, role-tags, preference-optimization, multi-turn, evaluation

## One-sentence takeaway
Role tags (user vs assistant) induce a measurable *source-of-truth bias* in modern instruction-tuned LLMs—most models overweight **user**-role claims when user and assistant statements conflict—and this bias can be amplified by preference alignment, reduced by reasoning tuning, and controllably steered with DPO.

## What problem does it solve?
- Sycophancy and “agreeing with the user” is often discussed behaviorally, but this paper isolates a *structural* driver: **role-tag conditioned training asymmetries** that make models treat information from different roles as having different credibility.
- Provides a task-agnostic way to quantify “who does the model believe when roles disagree?” which matters for multi-turn robustness under pressure.

## What is the core method / protocol?
- Define **user-assistant bias**: when the context contains conflicting information attributed to user vs assistant roles, measure whether the model preferentially relies on one role.
- Introduce **UserAssist** benchmark (task-agnostic) and evaluate **52 frontier models**.
- Controlled post-training ablations to attribute the bias to specific recipes:
  - human-preference alignment (RLHF-style) tends to amplify user bias
  - reasoning fine-tuning tends to reduce user bias
- Show bias can be **bidirectionally controlled** via **Direct Preference Optimization (DPO)** on UserAssist-train, and that the effect generalizes to a more realistic multi-turn conversation dataset.

## What are the key metrics?
- Primary: degree/direction of **user vs assistant reliance** under controlled conflicts (paper’s “user-assistant bias” score; exact formula not reproduced in the abstract).
- Secondary: generalization of the induced bias from UserAssist-train to a multi-turn conversation dataset.

## What are the main results?
- Most instruction-tuned assistants show **strong user bias**.
- Base models and “reasoning models” are closer to **neutral**.
- Alignment via human preferences increases user bias; reasoning FT decreases it.
- DPO can *increase or decrease* user bias and the change transfers beyond the synthetic benchmark to more realistic multi-turn conversations.

## How is this similar to GALILEO?
- Both focus on **multi-turn reliability under social/interaction pressure** and diagnosing failures where the model “goes along” with the user.
- The paper’s key axis (user vs assistant source weighting) is a plausible contributor to GALILEO-style drift/flip failures.

## How is this different from GALILEO?
- This is primarily about **role-tag / data-induced inductive bias** (source credibility asymmetry), not a full pressure ladder with time-to-failure and recovery dynamics.
- It’s closer to a *mechanism/diagnostic* for “agreement with user” than an end-to-end persuasion / drift-vs-revision evaluation suite.

## Where GALILEO is stronger / cleaner (if true)
- GALILEO can frame robustness failures as **trajectory phenomena** (when does it flip? does it recover? under what operator mix?), not only a static preference between two sources.
- GALILEO can more directly separate **evidence-driven revision** vs **pressure-driven drift** with explicit controls (if present in our design).

## Where GALILEO is weaker / needs to improve
- If GALILEO attributes “user agreement” mainly to social pressure, we may be missing a confound: a model might be “sycophantic” partly because it is **trained to trust the user role** (tag-induced), not (only) because of rhetorical pressure.
- GALILEO should consider role-tag effects explicitly, especially if prompts or evaluation harness systematically place misleading statements in the user role.

## Action items for GALILEO (experiments / method / writing)
- [ ] Add a **role-swap control**: for the same conflict/pressure content, swap whether the misleading claim is presented as *user* vs *assistant* (or “system quote”), and measure flips / ToF / recovery differences.
- [ ] Add a **source-credibility axis** to our analysis: “pressure content” vs “role-tag source” as separable factors.
- [ ] When reporting sycophancy/persuasion robustness, explicitly note whether the benchmark intrinsically tests **user-bias** (misinformation always injected via the user).
- [ ] Consider a mitigation baseline inspired by this work: preference optimization that penalizes blind user bias while preserving correct update behavior (conceptually similar to DuET-PD’s resist-vs-update balance).

## Quotes / details to potentially cite
- Abstract framing: “asymmetries in the training data associated with different role tags can introduce inductive biases.”
- Abstract result: “most of the instruction-tuned models exhibit strong user bias, whereas base and reasoning models are close to neutral.”
- Abstract attribution: “human-preference alignment amplifies user bias, while reasoning fine-tuning reduces it.”
- Abstract control: “user-assistant bias can be bidirectionally controlled via … DPO … and … generalizes to a more realistic multi-turn conversation dataset.”
