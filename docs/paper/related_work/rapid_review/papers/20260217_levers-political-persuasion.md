# The Levers of Political Persuasion with Conversational AI

- Year: 2025
- Venue: arXiv
- Authors: Kobi Hackenburg, Ben M. Tappin, Luke Hewitt, Ed Saunders, Sid Black, Hause Lin, Catherine Fist, Helen Margetts, David G. Rand, Christopher Summerfield
- URL: https://arxiv.org/abs/2507.13919
- BibTeX key (if we add it): hackenburg2025levers
- Tags: persuasion, political, dialogue, human-subjects, multi-turn

## One-sentence takeaway

Large-scale human-subject experiments suggest that *how* an LLM is post-trained/prompted matters more for political persuasiveness than raw scale/personalization—and that boosts in persuasiveness can come with systematic drops in factual accuracy.

## What problem does it solve?

- Quantifies the persuasive impact of conversational LLMs on political attitudes at scale, and decomposes which “levers” (model scale, prompting strategies, persuasive post-training) drive persuasion.
- Measures an important externality: whether increasing LLM persuasiveness is associated with reduced factual accuracy of claims made during persuasion.

## What is the core method / protocol?

- Three large-scale survey experiments with UK adults (total N=76,977).
- Participants have a **multi-turn chat** with one of **19 LLMs** about **1 of 707** politically balanced issues.
- Chat protocol: **2–10 turns**; treatment condition prompts the model to persuade the participant to a *pre-specified* stance.
- Outcomes:
  - Pre/post self-reported agreement on a percentage-point scale; persuasion effect computed vs a control group without the persuasive conversation.
  - Comparisons across:
    - Model scale (open-source and frontier models)
    - Prompting strategies (8 strategies; theory-motivated)
    - “Persuasive post-training” methods (3 methods; described as SFT / reward modelling / etc.)
  - Accuracy audit: counts and evaluates **466,769 fact-checkable claims** made across ~91k persuasive conversations, using a mix of LLM and professional human fact-checking.
- Validation sub-studies:
  - Static-message baseline: read a ~200-word persuasive message (no conversation) vs conversation.
  - Durability: follow-up one month later (study 1) to test persistence of attitude change.

## What are the key metrics?

- Persuasion effect: difference in mean post-treatment opinion between treatment vs control (reported in percentage points, pp).
- Relative persuasiveness comparisons (e.g., “+41% more persuasive than static message”).
- Factual accuracy of model claims (details of scoring are in the paper’s Methods/SM; reviewed here at the headline level).

## What are the main results?

- **Conversation beats static message** for persuasion:
  - GPT-4o: +2.94pp (reported as +41% more persuasive than static message).
  - GPT-4.5: +3.60pp (+52% more persuasive than static message).
- **Durability:** after one month, **~36–42%** of the immediate persuasion effect persisted (for GPT-4o in study 1’s follow-up).
- **Scale vs “levers”:** headline claim is that **post-training and prompting** increased persuasiveness more than personalization / increasing model scale, with effects up to **+51% (post-training)** and **+27% (prompting)**.
- **Accuracy tradeoff:** methods that increased persuasiveness also **systematically decreased factual accuracy** of claims during the persuasive conversations.

## How is this similar to GALILEO?

- Shares the theme that **multi-turn interaction dynamics** matter (conversational setting vs single-turn/static baselines).
- Explicitly studies **behavioral changes under pressure/optimization** (prompting/post-training) and surfaces a **tradeoff** between a targeted capability (persuasion) and an important safety/quality axis (factual accuracy), analogous to robustness tradeoffs GALILEO may want to measure.

## How is this different from GALILEO?

- Target is **human persuasion outcomes** on political attitudes (human-subjects), not LLM-to-LLM robustness or sycophancy/consistency per se.
- The dependent variable is **opinion shift** (human attitudes), whereas GALILEO is oriented toward **model behavioral stability/robustness** under multi-turn interaction (e.g., drift, agreement flips, failure time).
- Strongly domain-specific (political issues, UK adults), and focuses on persuader capability rather than model “truth maintenance” under user pressure.

## Where GALILEO is stronger / cleaner (if true)

- Can isolate **model-internal instability/flip dynamics** and evaluate with controlled protocols (including adversarial or pressure conditions) without confounds from human heterogeneity.
- Can more directly define/measure “robustness-to-pressure” outcomes (e.g., time-to-failure / recovery), whereas persuasion effects mix many human factors.

## Where GALILEO is weaker / needs to improve

- Lacks (or could benefit from) an explicit treatment of **externalities**: e.g., whether interventions that reduce sycophancy/increase robustness degrade other desiderata (helpfulness, engagement, perceived persuasiveness), mirroring this paper’s persuasion–accuracy tradeoff.
- Human-facing relevance: this paper makes a strong case that **conversational** settings can amplify effects; GALILEO should ensure its evaluation narratives emphasize multi-turn realism.

## Action items for GALILEO (experiments / method / writing)

- [ ] Add a short related-work paragraph framing multi-turn robustness as analogous to “conversation > static message” effects shown in large-scale persuasion studies.
- [ ] Consider adding a *tradeoff lens* in GALILEO: when we mitigate sycophancy/drift, do we change other axes (e.g., confidence, verbosity, or user satisfaction proxies)?
- [ ] If GALILEO includes interventions (prompting/finetuning), consider auditing whether those interventions shift **factuality** or **calibration**.

## Quotes / details to potentially cite

- Abstract (paraphrase-worthy): three experiments (N=76,977), 19 LLMs, 707 political issues; post-training (+51%) and prompting (+27%) boost persuasiveness more than personalization/scale; increased persuasiveness associated with decreased factual accuracy.
- Result summary (conversation vs static message): conversation was “+41%” (GPT-4o) and “+52%” (GPT-4.5) more persuasive than a static 200-word message.
- Durability: 36–42% of immediate persuasion effect persisted at one-month follow-up (study 1).
